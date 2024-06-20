import tensorflow as tf
import numpy as np
import string


class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = string.ascii_uppercase + 'UNKNOWN'
        if self.model.layers[0].input_shape[1] != 127:
            raise ValueError(f"Invalid model : {model_path}")
    
    def predict(self, landmarks_array):
        pred = self.model(np.expand_dims(landmarks_array, axis=0))
        cls = np.argmax(pred)
        return self.class_names[cls], pred[0][cls].numpy()


