from typing import Any, Generator

import numpy
import pyaudio
import torch
from silero_vad import load_silero_vad
from enum import Enum

from silero_vad.utils_vad import OnnxWrapper

class SamplingRate(Enum):
    LOW = 8000
    HIGH = 16000

    @classmethod
    def from_string(cls, value: str):
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

class VADRecorder:
    def __init__(self, sampling_rate: SamplingRate, channels: int):
        print("Initializing VADRecorder...")
        self.sampling_rate: SamplingRate = sampling_rate
        self.channels: int = channels
        self.model: OnnxWrapper = load_silero_vad()
        self.audio_processor = pyaudio.PyAudio()
        print("VADRecorder initialized with sampling rate {rate} and {channels} channels.".format(rate=sampling_rate.value, channels=channels))

    def start(self) -> Generator[tuple[bytes, float], Any, Any]:
        audio_stream: pyaudio.Stream = self.audio_processor.open(
            format=pyaudio.paInt16,
            channels=self.channels,
            rate=self.sampling_rate.value,
            input=True)

        while True:
            audio_chunk: pyaudio.paInt16 = audio_stream.read(self.sampling_rate.get_chunk_size())
            audio_data_array: numpy.ndarray = numpy.frombuffer(audio_chunk, numpy.int16).astype(numpy.float32) / 255.0

            current_confidence: float = self.model(torch.from_numpy(audio_data_array), self.sampling_rate.value).item()

            yield audio_chunk, current_confidence