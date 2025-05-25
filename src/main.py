import configparser
import numpy
from configparser import ConfigParser
from queue import Queue
from threading import Thread
from vad_recorder import VADRecorder, SamplingRate
from transcriber import Transcriber

def get_windowed_audio_chunks_in_order(audio_chunks: list[bytes], starting_index: int) -> bytes:
    """
    Returns the audio chunks in the order they were recorded, starting from the given index.
    """
    return b''.join(audio_chunks[starting_index:] + audio_chunks[:starting_index])

def calculate_window_size(rate: SamplingRate, window_ms: int) -> int:
    """
    Calculate the number of chunks needed for the evaluation window based on the sampling rate.
    """
    chunk_size_in_ms: int = int((rate.get_chunk_size() / rate.value) * 1000)
    return window_ms // chunk_size_in_ms

def transcribe_audio(transcriber: Transcriber, transcription_queue: Queue[bytes]) -> None:
    """
    Function to transcribe audio from the queue.
    This function is intended to be run in a separate thread.
    """
    # Placeholder for transcription logic
    while True:
        print(transcription_queue.get())
        transcription_queue.task_done()
        print("Transcribing audio")

def evaluate_audio(
        sampling_rate: SamplingRate,
        window_size: int,
        activation_threshold: float,
        deactivation_threshold: float,
        transcription_queue: Queue[bytes]) -> None:
    """
    Function to record audio using the VADRecorder.
    This function is intended to be run in a separate thread.
    """
    recorder: VADRecorder = VADRecorder(
        sampling_rate,
        parser.getint("AUDIO", "channels", fallback=1))
    collecting_audio: bool = False
    audio_to_transcribe: bytes = b''

    confidence_window: numpy.array = numpy.zeros(window_size)
    window_index: int = 0
    windowed_audio_chunks: list[bytes] = [bytes() for _ in range(window_size)]

    for audio, confidence in recorder.start():
        confidence_window[window_index] = confidence
        windowed_audio_chunks[window_index] = audio

        current_confidence = confidence_window.mean()

        if current_confidence > activation_threshold and not collecting_audio:
            print("Identified speech with {confidence} confidence. Start audio collection...".format(
                confidence=current_confidence))
            collecting_audio = True
            audio_to_transcribe = get_windowed_audio_chunks_in_order(windowed_audio_chunks, window_index)

        if confidence_window.mean() < deactivation_threshold and collecting_audio:
            print(
                "Identified end of speech with confidence of {confidence}. Stop collection and queue for transcription...".format(
                    confidence=current_confidence))
            transcription_queue.put(audio_to_transcribe)
            audio_to_transcribe = b''
            collecting_audio = False

        if collecting_audio:
            audio_to_transcribe += audio

        window_index = (window_index + 1) % window_size

if __name__ == "__main__":
    parser: configparser.ConfigParser = ConfigParser()
    parser.read("../config/config.ini")

    queue: Queue = Queue(maxsize=0)

    rate: SamplingRate = SamplingRate.from_string(parser.get("AUDIO", "sampling_rate", fallback="HIGH"))
    ws: int = calculate_window_size(rate, parser.getint("VOICE_ACTIVATION", "evaluation_window_ms"))
    act_threshold: float = parser.getfloat("VOICE_ACTIVATION", "activation_threshold", fallback=0.9)
    deact_threshold: float = parser.getfloat("VOICE_ACTIVATION", "deactivation_threshold", fallback=0.2)
    num_transcriber_threads: int = parser.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)

    trans: Transcriber = Transcriber(
        model_name=parser.get("TRANSCRIPTION", "model_name", fallback="tiny"),
        device_type=parser.get("TRANSCRIPTION", "device_type", fallback="cpu"),
        number_of_threads=parser.getint("TRANSCRIPTION", "number_of_whisper_threads", fallback=1))

    producer = Thread(target=evaluate_audio, args=(rate, ws, act_threshold, deact_threshold, queue))
    producer.start()

    threads = [producer]
    for _ in range(num_transcriber_threads):
        transcriber_thread = Thread(target=transcribe_audio, args=(trans, queue))
        threads.append(transcriber_thread)
        transcriber_thread.start()

    try:
        for thread in threads:
            thread.join()
    except KeyboardInterrupt:
        print("Exiting...")