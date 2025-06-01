from logging import Logger, getLogger
from queue import Queue, Empty
from threading import Event

from activation_word_detector import ActivationWordDetector
from speech_detector import SpeechDetector
from transcriber import Transcriber

LOGGER: Logger = getLogger(__name__)

def process_speech(
        command_transcriber: Transcriber,
        activation_word_detector: ActivationWordDetector,
        activation_detection_queue: Queue[bytes],
        command_transcription_queue: Queue[bytes],
        activation_detected_event: Event,
        command_processing_event: Event,
        cancellation_event: Event) -> None:
    """
    Function to process spoken word audio from the queue. First checks if the audio contains the activation word,
    and if the case it signals detection of the activation word.

    This function is intended to be run in a separate thread.

    :param command_transcriber: Transcriber instance to perform the transcription
    :param activation_word_detector: Detector to detect the activation word in a given audio sample
    :param activation_detection_queue: Queue to get live speech audio data for detection of the activation word
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param activation_detected_event: Event to signal that the activation word has been detected
    :param command_processing_event: Event to signal that the processing of commands is currently ongoing
    :param cancellation_event: Event to signal cancellation of the transcription process
    """
    while not cancellation_event.is_set():
        try:
            if not activation_detected_event.is_set():
                audio: bytes = activation_detection_queue.get(timeout=1)
                LOGGER.debug(f"Processing activation detection audio... (activation_detected_event={activation_detected_event.is_set()})")
                is_activation_word: bool = activation_word_detector.contains_activation_word(audio)
                if is_activation_word and not activation_detected_event.is_set():
                    _switch_from_detection_to_transcription(activation_detection_queue, activation_detected_event)
                    continue
            elif activation_detected_event.is_set() and not command_processing_event.is_set():
                audio: bytes = command_transcription_queue.get(timeout=1)
                command_processing_event.set()
                LOGGER.debug("Processing command audio...")
                servy_command: str = command_transcriber.transcribe(audio)
                LOGGER.info(f"Received command: {servy_command}")
                activation_detected_event.clear()
                command_processing_event.clear()
        except Empty:
            continue


def collect_speech(speech_detector: SpeechDetector,
                   activation_detection_queue: Queue[bytes],
                   command_transcription_queue: Queue[bytes],
                   activation_detected_event: Event,
                   command_processing_event: Event,
                   cancellation_event: Event) -> None:
    """
    Function to retrieve audio from the microphone using the VADRecorder, evaluate if there is speech,
    and queue the spoken word audio for further processing.

    This function is intended to be run in a separate thread.

    :param speech_detector: SpeechDetector instance to detect speech in audio feed
    :param activation_detection_queue: Queue to put live speech audio data for processing
    :param command_transcription_queue: Queue to put audio data to be transcribed as commands for the AI assistant
    :param activation_detected_event: Event to signal that the activation word has been detected
    :param command_processing_event: Event to signal that the processing of commands is currently ongoing
    :param cancellation_event: Event to signal cancellation of the detection process
    """
    for speech_chunk in speech_detector.start_collection(cancellation_event):
        if cancellation_event.is_set():
            break
        elif activation_detected_event.is_set() and not command_processing_event.is_set():
            LOGGER.debug(f"Activation detected and command not yet received -> putting chunk into command transcription queue")
            command_transcription_queue.put(speech_chunk)
            continue
        elif not activation_detected_event.is_set():
            activation_detection_queue.put(speech_chunk)


def _switch_from_detection_to_transcription(activation_detection_queue: Queue[bytes], activation_detected_event: Event) -> None:
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