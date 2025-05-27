import configparser
import threading
from configparser import ConfigParser
from queue import Queue, Empty
from threading import Thread

import numpy

from transcriber import Transcriber
from vad_recorder import VADRecorder, SamplingRate

def get_windowed_audio_chunks_in_order(audio_chunks: list[bytes], starting_index: int) -> bytes:
    """
    Returns the audio chunks in the order they were recorded, starting from the given index.
    :param audio_chunks: List of audio chunks
    :param starting_index: Index to start from
    """
    return b''.join(audio_chunks[starting_index:] + audio_chunks[:starting_index])

def calculate_window_size(rate: SamplingRate, window_ms: int) -> int:
    """
    Calculate the number of chunks needed for the evaluation window based on the sampling rate.
    :param rate: SamplingRate enum value (LOW or HIGH)
    :param window_ms: Window size in milliseconds
    """
    chunk_size_in_ms: int = int((rate.get_chunk_size() / rate.value) * 1000)
    return window_ms // chunk_size_in_ms

def transcribe_audio(transcriber: Transcriber, language: str, audio_to_transcribe_queue: Queue[bytes], cancellation_event: threading.Event) -> None:
    """
    Function to transcribe audio from the queue.
    This function is intended to be run in a separate thread.

    :param transcriber: Transcriber instance to perform the transcription
    :param language: Language code for the transcription (e.g., 'en' for English)
    :param audio_to_transcribe_queue: Queue containing audio data to be transcribed
    :param cancellation_event: Event to signal cancellation of the transcription process
    """

    while not cancellation_event.is_set():
        try:
            audio: bytes = audio_to_transcribe_queue.get(timeout=1)
        except Empty:
            continue
        transcribed_audio: str = transcriber.transcribe(
            audio,
            language=language,
            initial_prompt="Servy, Hey Servy, Hello Servy, Hi Servy, Please Servy, Hey Servy, Hi Servy, Hello Servy, Hey Survey, Hey Siri, Hey Server")
        print(transcribed_audio)

def process_live_audio(
        config_parser: ConfigParser,
        audio_to_transcribe_queue: Queue[bytes],
        cancellation_event: threading.Event) -> None:
    """
    Function to retrieve audio from the microphone using the VADRecorder, evaluate if there is speech,
    and queue the audio for transcription. When the keyword is detected, it starts collecting the
    audio servy command, transcribes it and publishes the resulting request.

    This function is intended to be run in a separate thread.

    :param config_parser: ConfigParser instance containing configuration settings
    :param audio_to_transcribe_queue: Queue to put audio data for transcription
    :param cancellation_event: Event to signal cancellation of the recording process
    """
    sampling_rate: SamplingRate = SamplingRate.from_string(config_parser.get("AUDIO", "sampling_rate", fallback="HIGH"))
    activation_window_size: int = calculate_window_size(sampling_rate, config_parser.getint("VOICE_ACTIVATION", "activation_window_ms"))
    deactivation_window_size: int = calculate_window_size(sampling_rate, config_parser.getint("VOICE_ACTIVATION", "deactivation_window_ms"))
    activation_threshold: float = config_parser.getfloat("VOICE_ACTIVATION", "activation_threshold", fallback=0.9)
    deactivation_threshold: float = config_parser.getfloat("VOICE_ACTIVATION", "deactivation_threshold", fallback=0.2)

    recorder: VADRecorder = VADRecorder(
        sampling_rate,
        parser.getint("AUDIO", "channels", fallback=1))
    collecting_audio: bool = False
    audio_to_transcribe: bytes = b''

    activation_confidence_window: numpy.array = numpy.zeros(activation_window_size)
    deactivation_confidence_window: numpy.array = numpy.zeros(deactivation_window_size)
    activation_window_index: int = 0
    deactivation_window_index: int = 0
    windowed_audio_chunks: list[bytes] = [bytes() for _ in range(activation_window_size)]

    for audio, confidence in recorder.start():
        if cancellation_event.is_set():
            break
        activation_confidence_window[activation_window_index] = confidence
        deactivation_confidence_window[deactivation_window_index] = confidence
        windowed_audio_chunks[activation_window_index] = audio

        current_confidence = activation_confidence_window.mean()

        if current_confidence > activation_threshold and not collecting_audio:
            print("Identified speech with {confidence} confidence. Start audio collection...".format(
                confidence=current_confidence))
            collecting_audio = True
            audio_to_transcribe = get_windowed_audio_chunks_in_order(windowed_audio_chunks, activation_window_index)

        if deactivation_confidence_window.mean() < deactivation_threshold and collecting_audio:
            print(
                "Identified end of speech with confidence of {confidence}. Stop collection and queue for transcription...".format(
                    confidence=current_confidence))
            audio_to_transcribe_queue.put(audio_to_transcribe)
            audio_to_transcribe = b''
            collecting_audio = False

        if collecting_audio:
            audio_to_transcribe += audio

        activation_window_index = (activation_window_index + 1) % activation_window_size
        deactivation_window_index = (deactivation_window_index + 1) % deactivation_window_size

if __name__ == "__main__":
    parser: configparser.ConfigParser = ConfigParser()
    parser.read("../config/config.ini")

    queue: Queue = Queue(maxsize=0)
    stop_event: threading.Event = threading.Event()

    producer = Thread(target=process_live_audio, args=(parser, queue, stop_event))
    producer.start()
    threads = [producer]

    audio_transcriber: Transcriber = Transcriber(
        model_name=parser.get("TRANSCRIPTION", "model_name", fallback="tiny"),
        device_type=parser.get("TRANSCRIPTION", "device_type", fallback="cpu"),
        number_of_threads=parser.getint("TRANSCRIPTION", "number_of_whisper_threads", fallback=1))
    configured_language: str = parser.getint("TRANSCRIPTION", "language", fallback="en")
    num_transcriber_threads: int = parser.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)
    for _ in range(num_transcriber_threads):
        transcriber_thread = Thread(target=transcribe_audio, args=(audio_transcriber, configured_language, queue, stop_event))
        threads.append(transcriber_thread)
        transcriber_thread.start()

    try:
        for thread in threads:
            thread.join()
    except (SystemExit, KeyboardInterrupt) as e:
        print("Exiting...")
        stop_event.set()
        for thread in threads:
            thread.join()