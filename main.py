import json
from argparse import ArgumentParser
from configparser import ConfigParser
from logging import Logger, basicConfig, getLogger
from queue import Queue
from threading import Thread, Event

from faster_whisper import WhisperModel
from openwakeword import Model
from pyaudio import PyAudio
from silero_vad import load_silero_vad

from activation_word_detector import ActivationWordDetector
from processing import collect_speech, process_speech
from speech_detector import SpeechDetector, SamplingRate
from transcriber import Transcriber
from utils import get_log_level

LOGGER: Logger = getLogger(__name__)

def _create_speech_detector(config: ConfigParser) -> SpeechDetector:
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


def _create_activation_word_detector(config: ConfigParser) -> ActivationWordDetector:
    """
    Creates and initializes an ActivationWordDetector instance based on the provided configuration.

    :param config: Configuration parser containing settings for the activation word detector
    """
    LOGGER.debug("Initializing activation detector...")
    model_paths: list[str] = json.loads(config.get("VOICE_ACTIVATION", "model_paths"))
    activation_model: Model = Model(wakeword_models=model_paths)
    detector: ActivationWordDetector = ActivationWordDetector(
        model=activation_model,
        activation_word_confidence_threshold=config.getfloat("VOICE_ACTIVATION", "activation_word_confidence_threshold",
                                                             fallback=0.5))
    LOGGER.debug("Finished initializing activation detector")
    return detector


def _create_command_transcriber(config: ConfigParser) -> Transcriber:
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


def run_application(arguments: ArgumentParser(), config: ConfigParser) -> None:
    arguments.add_argument("--log-level", help="Set the log level for the application", type=str, default="INFO")
    arguments = arguments.parse_args()

    if arguments.log_level:
        basicConfig(level=get_log_level(arguments.log_level))

    LOGGER.info("Starting Servy...")
    config.read("./config.ini")

    vad_queue: Queue = Queue(maxsize=0)
    command_queue: Queue = Queue(maxsize=1)

    exit_application_event: Event = Event()
    activation_event: Event = Event()
    command_processing_event: Event = Event()

    speech_detector: SpeechDetector = _create_speech_detector(config)
    activation_detector: ActivationWordDetector = _create_activation_word_detector(config)
    command_transcriber: Transcriber = _create_command_transcriber(config)

    producer: Thread = Thread(
        target=collect_speech,
        args=(speech_detector, vad_queue, command_queue, activation_event, command_processing_event, exit_application_event))
    producer.start()
    threads: list[Thread] = [producer]

    num_transcriber_threads: int = config.getint("TRANSCRIPTION", "number_of_transcriber_threads", fallback=1)
    for _ in range(num_transcriber_threads):
        transcriber_thread: Thread = Thread(
            target=process_speech,
            args=(command_transcriber, activation_detector, vad_queue, command_queue, activation_event, command_processing_event, exit_application_event))
        threads.append(transcriber_thread)
        transcriber_thread.start()

    LOGGER.info("Servy is now running. Press Ctrl+C to exit.")

    try:
        for thread in threads:
            thread.join()
    except (SystemExit, KeyboardInterrupt) as e:
        print("Exiting...")
        exit_application_event.set()
        for thread in threads:
            thread.join()

if __name__ == "__main__":
    run_application(ArgumentParser(), ConfigParser())