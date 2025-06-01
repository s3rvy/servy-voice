import threading
import unittest
from unittest.mock import MagicMock

from pyaudio import PyAudio, Stream
from silero_vad.utils_vad import OnnxWrapper
from torch import Tensor

from speech_detector import SpeechDetector, SamplingRate


class SpeechDetectorTests(unittest.TestCase):
    def test_start_collection_when_called_should_return_speech_from_mic_until_cancelled(self) -> None:
        mock_audio_processor: PyAudio = MagicMock(spec=PyAudio)
        audio_stream_mock = MagicMock(spec=Stream)
        mock_cancellation_event: threading.Event = MagicMock(spec=threading.Event)

        # Mock the audio data and VAD model responses:
        # begin of speech will be detected at chunk 4 and end detected at chunk 8
        # the cancellation event will be set at chunk 10 so the remaining positive detections will not be returned
        mock_vad_model: OnnxWrapper = MagicMock(spec=OnnxWrapper,
                                                side_effect=[Tensor([0.1]), Tensor([0.1]), Tensor([0.4]), Tensor([0.8]), Tensor([0.9]),
                                                             Tensor([0.9]), Tensor([0.9]), Tensor([0.0]), Tensor([0.9]), Tensor([0.9]),
                                                             Tensor([0.9]), Tensor([0.9]), Tensor([0.0]), Tensor([0.9]), Tensor([0.9])])
        audi_stream_function_mock: MagicMock = MagicMock(return_value=audio_stream_mock)
        read_function_mock: MagicMock = MagicMock(side_effect=[b'\x00' * 512, b'\x01' * 512, b'\x02' * 512, b'\x03' * 512, b'\x04' * 512,
                                                               b'\x05' * 512, b'\x06' * 512, b'\x07' * 512, b'\x08' * 512, b'\x09' * 512,
                                                               b'\x0a' * 512, b'\x0b' * 512, b'\x0c' * 512, b'\x0d' * 512, b'\x0e' * 512])
        mock_is_set_function: MagicMock = MagicMock(side_effect=[False, False, False, False, False, False, False, False, False, False, True])

        mock_audio_processor.open = audi_stream_function_mock
        audio_stream_mock.read = read_function_mock
        mock_cancellation_event.is_set = mock_is_set_function

        speech_detector = SpeechDetector(
            vad_model=mock_vad_model,
            audio_processor=mock_audio_processor,
            sampling_rate=SamplingRate.HIGH,
            activation_window_ms=100, # should lead to a window size of 3 chunks
            deactivation_window_ms=40, # should lead to a window size of 1 chunk
            activation_threshold=0.85,
            deactivation_threshold=0.1,
            channels=1
        )

        expected_audio: bytes = b''.join(
            [b'\x03' * 512, b'\x04' * 512, b'\x05' * 512, b'\x06' * 512, b'\x07' * 512]
        )
        for audio in speech_detector.start_collection(mock_cancellation_event):
            self.assertIsInstance(audio, bytes)
            self.assertEqual(audio, expected_audio)