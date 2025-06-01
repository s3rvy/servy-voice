from logging import Logger, getLogger

from numpy import frombuffer, int16
from openwakeword import Model

LOGGER: Logger = getLogger(__name__)

class ActivationWordDetector:
    def __init__(self, model: Model, activation_word_confidence_threshold: float = 0.5):
        self.model: Model = model
        self.activation_word_confidence_threshold: float = activation_word_confidence_threshold

    def contains_activation_word(self, audio: bytes) -> bool:
        """
        Detects if the activation word of any model specified is present in the given audio file.

        :param audio: Audio data to check for the activation word
        """
        predictions: list[dict[str, float]] = self.model.predict_clip(frombuffer(audio, dtype=int16))
        for prediction in predictions:
            for model, prediction_score in prediction.items():
                LOGGER.debug(f"Model '{model}' prediction score: {prediction_score}")
                if prediction_score > self.activation_word_confidence_threshold:
                    LOGGER.info(f"Activation word for model '{model}' detected with confidence {prediction_score}.")
                    return True

        return False