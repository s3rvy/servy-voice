import unittest
import numpy

from utils import audio_to_float, get_log_level
from parameterized import parameterized
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL

class UtilsTests(unittest.TestCase):
    def test_audio_to_float_when_called_should_return_float32_numpy_array(self) -> None:
        # Create a sample audio byte array
        audio_bytes: bytes = (numpy.random.rand(10) * 255).astype(numpy.int16).tobytes()

        # Call the function
        result: numpy.ndarray = audio_to_float(audio_bytes)

        # Check the type and dtype of the result
        self.assertIsInstance(result, numpy.ndarray)
        self.assertEqual(result.dtype, numpy.float32)

        # Check the values are in the expected range
        self.assertTrue(numpy.all(result >= 0.0))
        self.assertTrue(numpy.all(result <=1.0))

    @parameterized.expand([
        ("DEBUG", DEBUG),
        ("INFO", INFO),
        ("WARNING", WARNING),
        ("ERROR", ERROR),
        ("CRITICAL", CRITICAL),
        ("UNKNOWN", INFO)  # Default to INFO for unknown log levels
    ])
    def test_get_log_level_when_called_with_log_level_as_string_should_return_correct_log_level(self, log_level: str, expected_level: int) -> None:
        self.assertEqual(get_log_level(log_level), expected_level)