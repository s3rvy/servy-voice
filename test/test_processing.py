import unittest
from queue import Queue
from threading import Event
from unittest.mock import MagicMock

from activation_word_detector import ActivationWordDetector
from processing import process_speech, collect_speech
from speech_detector import SpeechDetector
from transcriber import Transcriber
from parameterized import parameterized


class ProcessingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.command_transcriber_mock: Transcriber = MagicMock(spec=Transcriber)
        self.activation_word_detector_mock: ActivationWordDetector = MagicMock(spec=ActivationWordDetector)
        self.activation_detection_queue: Queue = Queue()
        self.command_transcription_queue: Queue = Queue()
        self.activation_detected_event_mock: Event = MagicMock(spec=Event)
        self.command_processing_event_mock: Event = MagicMock(spec=Event)
        self.cancellation_event_mock: Event = MagicMock(spec=Event)

    @parameterized.expand(
        [
            (False,),
            (True,)
        ]
    )
    def test_process_speech_when_called_during_activation_detection_should_check_for_activation(self, contains_activation_word: bool) -> None:
        cancellation_is_set_function_mock: MagicMock = MagicMock(side_effect=[False, True])
        activation_detected_is_set_function_mock: MagicMock = MagicMock(return_value=False)

        self.activation_detected_event_mock.is_set = activation_detected_is_set_function_mock
        self.activation_detected_event_mock.set = MagicMock()
        self.cancellation_event_mock.is_set = cancellation_is_set_function_mock
        self.activation_word_detector_mock.contains_activation_word = MagicMock(return_value=contains_activation_word)

        self.activation_detection_queue.put(b"test audio data")
        self.activation_detection_queue.put(b"more test audio data that should not be processed")

        process_speech(
            command_transcriber=self.command_transcriber_mock,
            activation_word_detector=self.activation_word_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        self.activation_word_detector_mock.contains_activation_word.assert_called_with(b"test audio data")
        if contains_activation_word:
            self.activation_detected_event_mock.set.assert_called_once()
            self.assertTrue(self.activation_detection_queue.empty())
        else:
            self.activation_detected_event_mock.set.assert_not_called()
            self.assertEqual(self.activation_detection_queue.qsize(), 1)


    @parameterized.expand(
        [
            (False,),
            (True,)
        ]
    )
    def test_process_speech_when_called_after_activation_detection_should_transcribe_command_if_not_already_processing(self, already_processing: bool) -> None:
        cancellation_is_set_function_mock: MagicMock = MagicMock(side_effect=[False, True])
        activation_detected_is_set_function_mock: MagicMock = MagicMock(return_value=True)

        self.activation_detected_event_mock.is_set = activation_detected_is_set_function_mock
        self.cancellation_event_mock.is_set = cancellation_is_set_function_mock
        self.command_processing_event_mock.set = MagicMock()
        self.command_processing_event_mock.is_set = MagicMock(return_value=already_processing)
        self.command_transcriber_mock.transcribe = MagicMock(return_value="Test audio data")

        self.command_transcription_queue.put(b"test audio data")

        process_speech(
            command_transcriber=self.command_transcriber_mock,
            activation_word_detector=self.activation_word_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        if not already_processing:
            self.command_transcriber_mock.transcribe.assert_called_with(b"test audio data")
            self.command_processing_event_mock.set.assert_called_once()
        else:
            self.command_transcriber_mock.transcribe.assert_not_called()
            self.command_processing_event_mock.set.assert_not_called()

    def test_process_speech_when_queues_are_empty_should_wait_for_data_and_retry(self) -> None:
        cancellation_is_set_function_mock: MagicMock = MagicMock(side_effect=[False, False, True])
        self.cancellation_event_mock.is_set = cancellation_is_set_function_mock
        self.activation_detected_event_mock.is_set = MagicMock(side_effect=[False, False, True])
        self.command_processing_event_mock.is_set = MagicMock(return_value=False)

        self.activation_word_detector_mock.contains_activation_word = MagicMock()
        self.command_transcriber_mock.transcribe = MagicMock()


        process_speech(
            command_transcriber=self.command_transcriber_mock,
            activation_word_detector=self.activation_word_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        self.activation_word_detector_mock.contains_activation_word.assert_not_called()
        self.command_transcriber_mock.transcribe.assert_not_called()


    @parameterized.expand(
        [
            (False,),
            (True,)
        ]
    )
    def test_collect_speech_when_called_during_activation_detection_should_collect_speech_for_activation_detection_queue(self, cancel_collection) -> None:
        speech_detector_mock: SpeechDetector = MagicMock(spec=SpeechDetector)
        start_collecting_function_mock: MagicMock = MagicMock(side_effect=[iter([b"test audio data", b"more test audio data"])])
        speech_detector_mock.start_collection = start_collecting_function_mock

        self.activation_detected_event_mock.is_set = MagicMock(return_value=False)
        self.cancellation_event_mock.is_set = MagicMock(side_effect=[False, cancel_collection])

        collect_speech(
            speech_detector=speech_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        self.assertEqual(self.activation_detection_queue.get_nowait(), b"test audio data")
        if not cancel_collection:
            self.assertEqual(self.activation_detection_queue.get_nowait(), b"more test audio data")
        self.assertTrue(self.activation_detection_queue.empty())
        self.assertTrue(self.command_transcription_queue.empty())

    def test_collect_speech_when_activation_detected_should_collect_speech_for_command_transcription_queue(self) -> None:
        speech_detector_mock: SpeechDetector = MagicMock(spec=SpeechDetector)
        start_collecting_function_mock: MagicMock = MagicMock(side_effect=[iter([b"test audio data"])])
        speech_detector_mock.start_collection = start_collecting_function_mock

        self.activation_detected_event_mock.is_set = MagicMock(return_value=True)
        self.cancellation_event_mock.is_set = MagicMock(return_value=False)
        self.command_processing_event_mock.is_set = MagicMock(return_value=False)

        collect_speech(
            speech_detector=speech_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        self.assertTrue(self.activation_detection_queue.empty())
        self.assertEqual(self.command_transcription_queue.get_nowait(), b"test audio data")

    def test_collect_speech_when_activation_detected_and_aleady_processing_should_skip_speech_collection(self) -> None:
        speech_detector_mock: SpeechDetector = MagicMock(spec=SpeechDetector)
        start_collecting_function_mock: MagicMock = MagicMock(side_effect=[iter([b"test audio data"])])
        speech_detector_mock.start_collection = start_collecting_function_mock

        self.activation_detected_event_mock.is_set = MagicMock(return_value=True)
        self.cancellation_event_mock.is_set = MagicMock(return_value=False)
        self.command_processing_event_mock.is_set = MagicMock(return_value=True)

        collect_speech(
            speech_detector=speech_detector_mock,
            activation_detection_queue=self.activation_detection_queue,
            command_transcription_queue=self.command_transcription_queue,
            activation_detected_event=self.activation_detected_event_mock,
            command_processing_event=self.command_processing_event_mock,
            cancellation_event=self.cancellation_event_mock
        )

        self.assertTrue(self.activation_detection_queue.empty())
        self.assertTrue(self.command_transcription_queue.empty())
