import numpy
from faster_whisper import WhisperModel

from utils import audio_to_float


class Transcriber:
    def __init__(self, whisper_model: WhisperModel) -> None:
        """
        Initialize the Transcriber with a WhisperModel instance.

        :param whisper_model: An instance of WhisperModel from the faster_whisper package
        """
        self.whisper = whisper_model

    def transcribe(self, audio: bytes, language: str = None) -> str:
        """
        Transcribe the given audio bytes to text.

        :param language: Language code for the transcription (e.g., 'en' for English)
        :param audio: Audio data in bytes format
        :return: Transcribed text
        """
        audio_data_array: numpy.ndarray = audio_to_float(audio)
        segments, _ = self.whisper.transcribe(audio_data_array, language=language, beam_size=5, vad_filter=False)

        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        return transcription
