import configparser
import threading
from configparser import ConfigParser
from queue import Queue, Empty
from threading import Thread

from transcriber import Transcriber
from speech_detector import SamplingRate, SpeechDetector

def process_speech(
        transcriber: Transcriber,
        speech_audio_to_process_queue: Queue[bytes],
        cancellation_event: threading.Event) -> None:
    """
    Function to process spoken word audio from the queue. First checks if the audio contains the activation word,
    and if the case it signals detection of the activation word.

    This function is intended to be run in a separate thread.

    :param transcriber: Transcriber instance to perform the transcription
    :param speech_audio_to_process_queue: Queue containing live speech audio data to be processed
    :param cancellation_event: Event to signal cancellation of the transcription process
    """
    while not cancellation_event.is_set():
        try:
            audio: bytes = speech_audio_to_process_queue.get(timeout=1)
        except Empty:
            continue

        transcribed_audio: str = transcriber.transcribe(audio)
        print(transcribed_audio)

def collect_speech(
        config_parser: ConfigParser,
        speech_audio_to_process_queue: Queue[bytes],
        speech_detector: SpeechDetector,
        cancellation_event: threading.Event) -> None:
    """
    Function to retrieve audio from the microphone using the VADRecorder, evaluate if there is speech,
    and queue the spoken word audio for further processing.

    This function is intended to be run in a separate thread.

    :param config_parser: ConfigParser instance containing configuration settings
    :param speech_audio_to_process_queue: Queue to put live speech audio data for processing
    :param speech_detector: SpeechDetector instance to detect speech in audio feed
    :param cancellation_event: Event to signal cancellation of the recording process
    """
    sampling_rate: SamplingRate = SamplingRate.from_string(config_parser.get("AUDIO", "sampling_rate", fallback="HIGH"))
    activation_window_ms: int = config_parser.getint("VOICE_ACTIVATION", "activation_window_ms")
    deactivation_window_ms: int = config_parser.getint("VOICE_ACTIVATION", "deactivation_window_ms")
    activation_threshold: float = config_parser.getfloat("VOICE_ACTIVATION", "activation_threshold", fallback=0.9)
    deactivation_threshold: float = config_parser.getfloat("VOICE_ACTIVATION", "deactivation_threshold", fallback=0.2)
    channels: int = parser.getint("AUDIO", "channels", fallback=1)

    for speech_chunk in speech_detector.start_collection(
            cancellation_event,
            sampling_rate=sampling_rate,
            activation_window_ms=activation_window_ms,
            deactivation_window_ms=deactivation_window_ms,
            activation_threshold=activation_threshold,
            deactivation_threshold=deactivation_threshold,
            channels=channels):
        if cancellation_event.is_set():
            break
        speech_audio_to_process_queue.put(speech_chunk)

if __name__ == "__main__":
    parser: configparser.ConfigParser = ConfigParser()
    parser.read("config.ini")

    queue: Queue = Queue(maxsize=0)
    stop_event: threading.Event = threading.Event()

    producer = Thread(target=collect_speech, args=(parser, queue, SpeechDetector(), stop_event))
    producer.start()
    threads = [producer]

    audio_transcriber: Transcriber = Transcriber(
        model_name=parser.get("TRANSCRIPTION", "model_name", fallback="tiny"),
        device_type=parser.get("TRANSCRIPTION", "device_type", fallback="cpu"),
        number_of_threads=parser.getint("TRANSCRIPTION", "number_of_whisper_threads", fallback=1))

    num_transcriber_threads: int = parser.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)
    for _ in range(num_transcriber_threads):
        transcriber_thread = Thread(target=process_speech, args=(audio_transcriber, queue, stop_event))
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