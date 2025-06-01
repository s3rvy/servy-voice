import logging
import threading
from enum import Enum
from typing import Any, Generator

import numpy
import pyaudio
import torch
from silero_vad.utils_vad import OnnxWrapper

from utils import audio_to_float

LOGGER: logging.Logger = logging.getLogger(__name__)

class SamplingRate(Enum):
    LOW = 8000
    HIGH = 16000

    @classmethod
    def from_string(cls, value: str):
        """
        Convert a string representation of a sampling rate to the corresponding SamplingRate enum value.
        :param value: str - The string representation of the sampling rate, either "LOW" or "HIGH".
        :return: SamplingRate - The corresponding SamplingRate enum value.
        """
        if value == "LOW":
            return cls.LOW
        elif value == "HIGH":
            return cls.HIGH
        else:
            raise ValueError("Invalid sampling rate {rate}. Only LOW (=8000) and HIGH (16000) are allowed".format(rate=value))

    def get_chunk_size(self):
        if self == SamplingRate.LOW:
            return 256
        elif self == SamplingRate.HIGH:
            return 512
        else:
            raise ValueError("Invalid sampling rate")

class SpeechDetector:
    def __init__(
            self,
            vad_model: OnnxWrapper,
            audio_processor: pyaudio.PyAudio,
            sampling_rate: SamplingRate = SamplingRate.HIGH,
            activation_window_ms: int = 400,
            deactivation_window_ms: int = 200,
            activation_threshold: float = 0.85,
            deactivation_threshold: float = 0.1,
            channels: int = 1):
        """
        Initialize the VADRecorder with the specified parameters.

        :param sampling_rate: SamplingRate enum value (LOW or HIGH)
        :param activation_window_ms: Size of the activation window in milliseconds
        :param deactivation_window_ms: Size of the deactivation window in milliseconds
        :param activation_threshold: Threshold for activating speech detection
        :param deactivation_threshold: Threshold for deactivating speech detection
        :param channels: Number of audio channels (1 for mono, 2 for stereo)
        """
        self._vad_model: OnnxWrapper = vad_model
        self._audio_processor = audio_processor
        self._sampling_rate: SamplingRate = sampling_rate
        self._activation_window_size: int = _calculate_window_size(self._sampling_rate, activation_window_ms)
        self._deactivation_window_size: int = _calculate_window_size(self._sampling_rate, deactivation_window_ms)
        self._activation_threshold: float = activation_threshold
        self._deactivation_threshold: float = deactivation_threshold
        self._channels: int = channels

    def start_collection(
            self,
            cancellation_event: threading.Event) -> Generator[bytes, Any, Any]:
        """
        This method continuously reads audio data from the microphone, processes it with the VAD model,
        and analyses the confidence of speech detection. If speech is detected in a sliding window of activation_window_ms
        above the activation threshold, audio recording starts including the already detected audio in the window. When
        the confidence drops below the deactivation threshold for a window length of deactivation_window_ms,
        the collected audio is yielded as one continuous byte chunk.

        :param cancellation_event: Event to signal cancellation of the detection process

        :yield: A chunk of audio data (bytes) which has been detected as speech.
        """
        audio_stream: pyaudio.Stream = self._audio_processor.open(
            format=pyaudio.paInt16,
            channels=self._channels,
            rate=self._sampling_rate.value,
            input=True)
        collecting_audio: bool = False
        audio_to_transcribe: bytes = b''

        activation_confidence_window: numpy.array = numpy.zeros(self._activation_window_size)
        deactivation_confidence_window: numpy.array = numpy.zeros(self._deactivation_window_size)
        activation_window_index: int = 0
        deactivation_window_index: int = 0
        windowed_audio_chunks: list[bytes] = [bytes() for _ in range(self._activation_window_size)]

        while not cancellation_event.is_set():
            audio_chunk: pyaudio.paInt16 = audio_stream.read(self._sampling_rate.get_chunk_size(), exception_on_overflow=False)
            audio_data_array: numpy.ndarray = audio_to_float(audio_chunk)

            confidence: float = self._vad_model(torch.from_numpy(audio_data_array), self._sampling_rate.value).item()

            activation_confidence_window[activation_window_index] = confidence
            deactivation_confidence_window[deactivation_window_index] = confidence
            windowed_audio_chunks[activation_window_index] = audio_chunk

            activation_window_index = (activation_window_index + 1) % self._activation_window_size
            deactivation_window_index = (deactivation_window_index + 1) % self._deactivation_window_size

            current_confidence = activation_confidence_window.mean()

            if collecting_audio:
                audio_to_transcribe += audio_chunk

            if current_confidence > self._activation_threshold and not collecting_audio:
                LOGGER.info("Identified speech with {confidence} confidence. Start audio collection...".format(
                    confidence=current_confidence))
                collecting_audio = True
                audio_to_transcribe = _get_windowed_audio_chunks_in_order(windowed_audio_chunks,
                                                                          activation_window_index)

            if deactivation_confidence_window.mean() < self._deactivation_threshold and collecting_audio:
                LOGGER.info(
                    "Identified end of speech with confidence of {confidence}. Stop collection and queue for transcription...".format(
                        confidence=current_confidence))
                yield audio_to_transcribe
                audio_to_transcribe = b''
                collecting_audio = False

def _get_windowed_audio_chunks_in_order(audio_chunks: list[bytes], starting_index: int) -> bytes:
    """
    Returns the audio chunks in the order they were recorded, starting from the given index.
    :param audio_chunks: List of audio chunks
    :param starting_index: Index to start from
    """
    return b''.join(audio_chunks[starting_index:] + audio_chunks[:starting_index])

def _calculate_window_size(rate: SamplingRate, window_ms: int) -> int:
    """
    Calculate the number of chunks needed for the evaluation window based on the sampling rate.
    :param rate: SamplingRate enum value (LOW or HIGH)
    :param window_ms: Window size in milliseconds
    """
    chunk_size_in_ms: int = int((rate.get_chunk_size() / rate.value) * 1000)
    return window_ms // chunk_size_in_ms

