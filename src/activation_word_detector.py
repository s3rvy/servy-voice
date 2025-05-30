import numpy
import openwakeword

class ActivationWordDetector:
    def __init__(self, model_paths: list[str], activation_word_confidence_threshold: float = 0.5):
        self.model: openwakeword.Model = openwakeword.Model(wakeword_models=model_paths)
        self.activation_word_confidence_threshold: float = activation_word_confidence_threshold

    def contains_activation_word(self, audio: bytes) -> bool:
        """
        Detects if the activation word of any model specified is present in the given audio file.

        :param audio: Audio data to check for the activation word
        """
        predictions: list[dict[str, float]] = self.model.predict_clip(numpy.frombuffer(audio, dtype=numpy.int16))
        for prediction in predictions:
            for word, prediction_score in prediction.items():
                if prediction_score > self.activation_word_confidence_threshold:
                    print(f"Activation word '{word}' detected with confidence {prediction_score}.")
                    return True

        return False