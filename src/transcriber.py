import numpy
from faster_whisper import WhisperModel

from utils import audio_to_float


class Transcriber:
    def __init__(self, model_name: str, device_type: str, number_of_threads: int):
        print("Initializing Transcriber...")
        self.whisper = WhisperModel(
            model_name,
            device=device_type,
            compute_type="int8",
            cpu_threads=number_of_threads,
            local_files_only=False)
        print("Transcriber initialized")

    def transcribe(self, audio: bytes, language: str) -> str:
        """
        Transcribe the given audio bytes to text.

        :param language: Language code for the transcription (e.g., 'en' for English)
        :param audio: Audio data in bytes format
        :return: Transcribed text
        """
        audio_data_array: numpy.ndarray = audio_to_float(audio)
        segments, _ = self.whisper.transcribe(audio_data_array,
                                language=language,
                                beam_size=5,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=1000))

        segments = [s.text for s in segments]
        transcription = " ".join(segments)
        return transcription
