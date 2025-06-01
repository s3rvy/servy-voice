from numpy import frombuffer, int16, float32, ndarray
from logging import DEBUG, INFO, WARNING, ERROR, CRITICAL


def audio_to_float(audio: bytes) -> ndarray:
    """
    Convert audio bytes to a float32 numpy array.

    :param audio: Audio data in bytes format
    :return: Numpy array of float32 values
    """
    return frombuffer(audio, dtype=int16).astype(float32) / 255.0


def get_log_level(log_level: str) -> int:
    """
    Converts a string representation of a log level to the corresponding logging level constant.

    :param log_level: String representation of the log level (e.g., "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
    :return: Corresponding logging level constant
    """
    match log_level:
        case "DEBUG":
            return DEBUG
        case "INFO":
            return INFO
        case "WARNING":
            return WARNING
        case "ERROR":
            return ERROR
        case "CRITICAL":
            return CRITICAL
        case _:
            return INFO