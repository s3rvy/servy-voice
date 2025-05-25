from faster_whisper import WhisperModel
from configparser import ConfigParser

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