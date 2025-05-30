import configparser
import json
import threading
from configparser import ConfigParser
from queue import Queue, Empty
from threading import Thread

from activation_word_detector import ActivationWordDetector
from speech_detector import SamplingRate, SpeechDetector
from transcriber import Transcriber


def process_speech(
        transcriber: Transcriber,
        activation_word_detector: ActivationWordDetector,
        activation_detection_queue: Queue[bytes],
        command_transcription_queue: Queue[bytes],
        activation_detected_event: threading.Event,
        cancellation_event: threading.Event) -> None:
    """
    Function to process spoken word audio from the queue. First checks if the audio contains the activation word,
    and if the case it signals detection of the activation word.

    This function is intended to be run in a separate thread.

    :param transcriber: Transcriber instance to perform the transcription
    :param activation_word_detector: Detector to detect the activation word in a given audio sample
    :param activation_detection_queue: Queue to get live speech audio data for detection of the activation word
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param activation_detected_event: Event to signal that the activation word has been detected
    :param cancellation_event: Event to signal cancellation of the transcription process
    """
    while not cancellation_event.is_set():
        try:
            if not activation_detected_event.is_set():
                audio: bytes = activation_detection_queue.get(timeout=1)
                print(
                    f"Processing activation detection audio... (activation_detected_event={activation_detected_event.is_set()})")
                is_activation_word: bool = activation_word_detector.contains_activation_word(audio)
                if is_activation_word and not activation_detected_event.is_set():
                    switch_from_detection_to_transcription(activation_detection_queue, activation_detected_event)
                    continue
            else:
                audio: bytes = command_transcription_queue.get(timeout=1)
                print("Processing command audio...")
                servy_command: str = transcriber.transcribe(audio)
                print(f"Got command: {servy_command}")
                activation_detected_event.clear()
        except Empty:
            continue

def switch_from_detection_to_transcription(activation_detection_queue: Queue[bytes], activation_detected_event: threading.Event) -> None:
    """
    Switches from activation detection to command transcription by setting the activation_detected_event
    and clearing the activation detection queue.

    :param activation_detection_queue: Queue to clear the activation detection audio data
    """
    print("Activation word detected! Set activation_detected_event and clear activation detection queue.")
    activation_detected_event.set()
    with activation_detection_queue.mutex:
        activation_detection_queue.queue.clear()
        activation_detection_queue.unfinished_tasks = 0

def collect_speech(
        config_parser: ConfigParser,
        speech_detector: SpeechDetector,
        activation_detection_queue: Queue[bytes],
        command_transcription_queue: Queue[bytes],
        activation_detected_event: threading.Event,
        cancellation_event: threading.Event) -> None:
    """
    Function to retrieve audio from the microphone using the VADRecorder, evaluate if there is speech,
    and queue the spoken word audio for further processing.

    This function is intended to be run in a separate thread.

    :param config_parser: ConfigParser instance containing configuration settings
    :param speech_detector: SpeechDetector instance to detect speech in audio feed
    :param activation_detection_queue: Queue to put live speech audio data for processing
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param cancellation_event: Event to signal cancellation of the detection process
    :param activation_detected_event: Event to signal that the activation word has been detected
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
        elif activation_detected_event.is_set() and not command_transcription_queue.full():
            print(f"Activation detected and command not yet received -> putting chunk into command transcription queue")
            command_transcription_queue.put(speech_chunk)
            continue
        elif not activation_detected_event.is_set():
            activation_detection_queue.put(speech_chunk)
        
def join_all(threads_to_join: list[Thread]) -> None:
    """
    Joins all threads in the provided list.

    :param threads_to_join: List of threads to join
    """
    for thread in threads_to_join:
        thread.join()

if __name__ == "__main__":
    parser: configparser.ConfigParser = ConfigParser()
    parser.read("config.ini")

    vad_queue: Queue = Queue(maxsize=0)
    command_queue: Queue = Queue(maxsize=1)
    exit_application_event: threading.Event = threading.Event()
    activation_event: threading.Event = threading.Event()

    audio_transcriber: Transcriber = Transcriber(
        model_name=parser.get("TRANSCRIPTION", "model_name", fallback="tiny"),
        device_type=parser.get("TRANSCRIPTION", "device_type", fallback="cpu"),
        number_of_threads=parser.getint("TRANSCRIPTION", "number_of_whisper_threads", fallback=1))
    model_paths: list[str] = json.loads(parser.get("VOICE_ACTIVATION", "model_paths"))
    activation_detector: ActivationWordDetector = ActivationWordDetector(
        model_paths=model_paths,
        activation_word_confidence_threshold=parser.getfloat("VOICE_ACTIVATION", "activation_word_confidence_threshold", fallback=0.5))

    producer = Thread(target=collect_speech, args=(parser, SpeechDetector(), vad_queue, command_queue, activation_event, exit_application_event))
    producer.start()
    threads = [producer]

    num_transcriber_threads: int = parser.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)
    for _ in range(num_transcriber_threads):
        transcriber_thread = Thread(target=process_speech, args=(audio_transcriber, activation_detector, vad_queue, command_queue, activation_event, exit_application_event))
        threads.append(transcriber_thread)
        transcriber_thread.start()

    try:
        join_all(threads)
    except (SystemExit, KeyboardInterrupt) as e:
        print("Exiting...")
        exit_application_event.set()
        join_all(threads)
            