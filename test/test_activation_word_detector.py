import unittest
from unittest.mock import MagicMock

import numpy
from openwakeword import Model

from activation_word_detector import ActivationWordDetector


class ActivationWordDetectorTests(unittest.TestCase):
    def test_contains_activation_word_when_called_with_audio_containing_activation_world_should_return_true(self) -> None:
        model_mock: Model = MagicMock(spec=Model)
        activation_word_detector: ActivationWordDetector = ActivationWordDetector(model_mock)
        function_mock: MagicMock = MagicMock(return_value=[
            {"model1": 0.2, "model2": 0.4},
            {"model1": 0.3, "model2": 0.7},
            {"model1": 0.8, "model2": 0.2}
        ])
        model_mock.predict_clip = function_mock

        audio_bytes: bytes = (numpy.random.rand(10)).astype(numpy.int16).tobytes()

        result: bool = activation_word_detector.contains_activation_word(audio_bytes)

        self.assertTrue(result)
        numpy.testing.assert_array_equal(function_mock.call_args[0][0], numpy.frombuffer(audio_bytes, dtype=numpy.int16))

    def test_contains_activation_word_when_called_with_audio_not_containing_activation_world_should_return_false(self) -> None:
        model_mock: Model = MagicMock(spec=Model)
        activation_word_detector: ActivationWordDetector = ActivationWordDetector(model_mock)
        function_mock: MagicMock = MagicMock(return_value=[
            {"model1": 0.0, "model2": 0.0},
            {"model1": 0.0, "model2": 0.0},
            {"model1": 0.0, "model2": 0.0}
        ])
        model_mock.predict_clip = function_mock

        audio_bytes: bytes = (numpy.random.rand(10)).astype(numpy.int16).tobytes()

        result: bool = activation_word_detector.contains_activation_word(audio_bytes)

        self.assertFalse(result)
        numpy.testing.assert_array_equal(function_mock.call_args[0][0], numpy.frombuffer(audio_bytes, dtype=numpy.int16))