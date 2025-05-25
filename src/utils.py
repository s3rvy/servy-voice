import numpy

def audio_to_float(audio: bytes) -> numpy.ndarray:
    """
    Convert audio bytes to a float32 numpy array.

    :param audio: Audio data in bytes format
    :return: Numpy array of float32 values
    """
    return numpy.frombuffer(audio, dtype=numpy.int16).astype(numpy.float32) / 255.0