import unittest
from unittest.mock import MagicMock

import numpy
from faster_whisper import WhisperModel
from faster_whisper.transcribe import Segment

from transcriber import Transcriber
from utils import audio_to_float

from parameterized import parameterized

class TranscriberTests(unittest.TestCase):
    @parameterized.expand(
        [
            ("Hello world", "Hello world", None),
            ("Another example", "Another example", "en")
        ]
    )
    def test_transcribe_when_called_should_return_transcription(self, spoken_audio: str, expected_transcription: str, language: str) -> None:
        model_mock: WhisperModel = MagicMock(spec=WhisperModel)
        transcriber: Transcriber = Transcriber(model_mock)
        segment_mock: Segment = MagicMock()
        segment_mock.text = spoken_audio
        function_mock: MagicMock = MagicMock(return_value=([segment_mock], None))
        model_mock.transcribe = function_mock

        audio_bytes = (numpy.random.rand(10) * 255).astype(numpy.int16).tobytes()

        result: str = transcriber.transcribe(audio_bytes, language=language)

        self.assertTrue(result == expected_transcription)
        numpy.testing.assert_array_equal(function_mock.call_args[0][0], audio_to_float(audio_bytes))
        self.assertEqual(function_mock.call_args[1]["language"], language)