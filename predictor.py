import tensorflow as tf
import numpy as np
import string


class Predictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.class_names = string.ascii_uppercase + 'UNKNOWN'
        if self.model.layers[0].input_shape[1] != 127:
            raise ValueError(f"Invalid model : {model_path}")
    
    def predict(self, mp_result):
        landmarks_list = []
        landmarks_list.append(mp_result.multi_handedness[0].classification[0].index)

        hl_wl = list(mp_result.multi_hand_landmarks[0].landmark) + \
                list(mp_result.multi_hand_world_landmarks[0].landmark)
        for landmark in hl_wl:
            landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks_array = np.array([landmarks_list], dtype=np.float32)

        pred = self.model(landmarks_array)
        cls = np.argmax(pred)
        return self.class_names[cls], pred[0][cls].numpy()


