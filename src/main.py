import configparser
import json
import logging
import threading
from argparse import ArgumentParser
from configparser import ConfigParser
from logging import Logger
from queue import Queue, Empty
from threading import Thread

import openwakeword
from faster_whisper import WhisperModel
from pyaudio import PyAudio

from activation_word_detector import ActivationWordDetector
from speech_detector import SamplingRate, SpeechDetector
from transcriber import Transcriber
from silero_vad import load_silero_vad

def get_log_level(log_level: str) -> int:
    """
    Converts a string representation of a log level to the corresponding logging level constant.

    :param log_level: String representation of the log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    :return: Corresponding logging level constant
    """
    match log_level:
        case "DEBUG":
            return logging.DEBUG
        case "INFO":
            return logging.INFO
        case "WARNING":
            return logging.WARNING
        case "ERROR":
            return logging.ERROR
        case "CRITICAL":
            return logging.CRITICAL
        case _:
            return logging.INFO

def process_speech(
        transcriber: Transcriber,
        detector: ActivationWordDetector,
        activation_detection_queue: Queue[bytes],
        command_transcription_queue: Queue[bytes],
        activation_detected_event: threading.Event,
        processing_event: threading.Event,
        cancellation_event: threading.Event) -> None:
    """
    Function to process spoken word audio from the queue. First checks if the audio contains the activation word,
    and if the case it signals detection of the activation word.

    This function is intended to be run in a separate thread.

    :param transcriber: Transcriber instance to perform the transcription
    :param detector: Detector to detect the activation word in a given audio sample
    :param activation_detection_queue: Queue to get live speech audio data for detection of the activation word
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param activation_detected_event: Event to signal that the activation word has been detected
    :param processing_event: Event to signal that the processing of commands is currently ongoing
    :param cancellation_event: Event to signal cancellation of the transcription process
    """
    while not cancellation_event.is_set():
        try:
            if not activation_detected_event.is_set():
                audio: bytes = activation_detection_queue.get(timeout=1)
                LOGGER.debug(f"Processing activation detection audio... (activation_detected_event={activation_detected_event.is_set()})")
                is_activation_word: bool = detector.contains_activation_word(audio)
                if is_activation_word and not activation_detected_event.is_set():
                    switch_from_detection_to_transcription(activation_detection_queue, activation_detected_event)
                    continue
            else:
                audio: bytes = command_transcription_queue.get(timeout=1)
                processing_event.set()
                LOGGER.debug("Processing command audio...")
                servy_command: str = transcriber.transcribe(audio)
                LOGGER.info(f"Received command: {servy_command}")
                activation_detected_event.clear()
                processing_event.clear()
        except Empty:
            continue

def switch_from_detection_to_transcription(activation_detection_queue: Queue[bytes], activation_detected_event: threading.Event) -> None:
    """
    Switches from activation detection to command transcription by setting the activation_detected_event
    and clearing the activation detection queue.

    :param activation_detection_queue: Queue to clear the activation detection audio data
    :param activation_detected_event: Event to signal that the activation word has been detected
    """
    LOGGER.info("Activation word detected! Set activation_detected_event and clear activation detection queue.")
    activation_detected_event.set()
    with activation_detection_queue.mutex:
        activation_detection_queue.queue.clear()
        activation_detection_queue.unfinished_tasks = 0

def collect_speech(detector: SpeechDetector,
        activation_detection_queue: Queue[bytes],
        command_transcription_queue: Queue[bytes],
        activation_detected_event: threading.Event,
        processing_event: threading.Event,
        cancellation_event: threading.Event) -> None:
    """
    Function to retrieve audio from the microphone using the VADRecorder, evaluate if there is speech,
    and queue the spoken word audio for further processing.

    This function is intended to be run in a separate thread.

    :param detector: SpeechDetector instance to detect speech in audio feed
    :param activation_detection_queue: Queue to put live speech audio data for processing
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param activation_detected_event: Event to signal that the activation word has been detected
    :param processing_event: Event to signal that the processing of commands is currently ongoing
    :param cancellation_event: Event to signal cancellation of the detection process
    """

    for speech_chunk in detector.start_collection(cancellation_event):
        if cancellation_event.is_set():
            break
        elif activation_detected_event.is_set() and not (command_transcription_queue.full() or processing_event.is_set()):
            LOGGER.debug(f"Activation detected and command not yet received -> putting chunk into command transcription queue")
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

def create_speech_detector(
        config: configparser.ConfigParser) -> SpeechDetector:
    """
    Creates and initializes a SpeechDetector instance based on the provided configuration.

    :param config: Configuration parser containing settings for the speech detector
    """
    LOGGER.debug("Initializing speech detector...")
    detector: SpeechDetector = SpeechDetector(
        load_silero_vad(),
        PyAudio(),
        sampling_rate=SamplingRate.from_string(config.get("AUDIO", "sampling_rate", fallback="HIGH")),
        activation_window_ms=config.getint("VOICE_ACTIVATION", "activation_window_ms"),
        deactivation_window_ms=config.getint("VOICE_ACTIVATION", "deactivation_window_ms"),
        activation_threshold=config.getfloat("VOICE_ACTIVATION", "activation_threshold", fallback=0.9),
        deactivation_threshold=config.getfloat("VOICE_ACTIVATION", "deactivation_threshold", fallback=0.2),
        channels=config.getint("AUDIO", "channels", fallback=1))
    LOGGER.debug("Finished initializing speech detector")
    return detector

def create_activation_word_detector(
        config: configparser.ConfigParser) -> ActivationWordDetector:
    """
    Creates and initializes an ActivationWordDetector instance based on the provided configuration.

    :param config: Configuration parser containing settings for the activation word detector
    """
    LOGGER.debug("Initializing activation detector...")
    model_paths: list[str] = json.loads(config.get("VOICE_ACTIVATION", "model_paths"))
    activation_model: openwakeword.Model = openwakeword.Model(wakeword_models=model_paths)
    detector: ActivationWordDetector = ActivationWordDetector(
        model=activation_model,
        activation_word_confidence_threshold=config.getfloat("VOICE_ACTIVATION", "activation_word_confidence_threshold", fallback=0.5))
    LOGGER.debug("Finished initializing activation detector")
    return detector

def create_command_transcriber(config: configparser.ConfigParser) -> Transcriber:
    """
    Creates and initializes a Transcriber instance based on the provided configuration.

    :param config: Configuration parser containing settings for the transcriber
    """
    LOGGER.debug("Initializing Transcriber...")
    device_type: str = config.get("TRANSCRIPTION", "device_type", fallback="cpu")
    whisper_model: WhisperModel = WhisperModel(
            model_size_or_path=config.get("TRANSCRIPTION", "model_name", fallback="tiny"),
            device=device_type,
            compute_type="float16" if device_type != "cpu" else "int8",
            cpu_threads=config.getint("TRANSCRIPTION", "number_of_whisper_threads", fallback=1),
            local_files_only=False)
    transcriber: Transcriber = Transcriber(whisper_model)
    LOGGER.debug("Finished initializing transcriber")
    return transcriber

def create_logger(argument_parser: ArgumentParser) -> Logger:
    """
    Creates a logger for the application based on command line arguments.
    """
    argument_parser.add_argument("--log-level", help="Set the log level for the application", type=str, default="INFO")
    arguments = argument_parser.parse_args()

    if arguments.log_level:
        logging.basicConfig(level=get_log_level(arguments.log_level))
    return logging.getLogger(__name__)

LOGGER: Logger = create_logger(ArgumentParser())

if __name__ == "__main__":
    LOGGER.info("Starting Servy...")
    parser: configparser.ConfigParser = ConfigParser()
    parser.read("config.ini")

    vad_queue: Queue = Queue(maxsize=0)
    command_queue: Queue = Queue(maxsize=1)

    exit_application_event: threading.Event = threading.Event()
    activation_event: threading.Event = threading.Event()
    command_processing_event: threading.Event = threading.Event()

    speech_detector: SpeechDetector = create_speech_detector(parser)
    activation_detector: ActivationWordDetector = create_activation_word_detector(parser)
    command_transcriber: Transcriber = create_command_transcriber(parser)

    producer = Thread(
        target=collect_speech,
        args=(speech_detector, vad_queue, command_queue, activation_event, command_processing_event, exit_application_event))
    producer.start()
    threads = [producer]

    num_transcriber_threads: int = parser.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)
    for _ in range(num_transcriber_threads):
        transcriber_thread = Thread(
            target=process_speech,
            args=(command_transcriber, activation_detector, vad_queue, command_queue, activation_event, command_processing_event, exit_application_event))
        threads.append(transcriber_thread)
        transcriber_thread.start()

    LOGGER.info("Servy is now running. Press Ctrl+C to exit.")

    try:
        join_all(threads)
    except (SystemExit, KeyboardInterrupt) as e:
        print("Exiting...")
        exit_application_event.set()
        join_all(threads)
            