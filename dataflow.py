import mediapipe as mp
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import atexit
import string

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

detector = mp_hands.Hands(model_complexity=0,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5)

atexit.register(detector.close)

@tf.py_function(Tout=tf.uint8)
def load_image(image_path):
    """
    Load an image and convert it to a tensor.
    
    Args:
        image_path (str): Path to the image file.

    Returns:
        tf.Tensor: Image tensor with dtype <uint8>.
    """
    img = tf.io.read_file(image_path)
    img = tf.io.decode_image(img)
    return img


def generate_image_label_pairs(directory, class_names):
    """
    Generate image paths and labels.
    
    Args:
        directory (str): Path to the directory containing images.
        class_names (list): List of class names.
        
    Yields:
        tuple: (image_path, label)
    """
    class_indices = {class_name: index for index, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(directory, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                label = class_indices[class_name]
                yield image_path, label


def image_dataset_from_directory(directory, class_names=None, shuffle=True, seed=42):
    """
    Create a tf.data.Dataset from a directory of images.

    Args:
        directory (str): Path to the directory containing images.
        class_names (list): List of class names. If None, class names will be inferred from subdirectory names.
        shuffle (bool): Whether to shuffle the dataset.
        seed (int): Random seed for shuffling.

    Returns:
        tf.data.Dataset: Dataset yielding (image, label) pairs.
    """

    if class_names is None:
        class_names = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    # Create a dataset of image paths and labels
    image_label_pairs = list(generate_image_label_pairs(directory, class_names))
    image_paths, labels = zip(*image_label_pairs)

    # Create a tf.data.Dataset from the image paths and labels
    path_ds = tf.data.Dataset.from_tensor_slices(list(image_paths))
    label_ds = tf.data.Dataset.from_tensor_slices(list(labels))

    # Map the image paths to actual images
    image_ds = path_ds.map(load_image)

    # Zip the image and label datasets together
    dataset = tf.data.Dataset.zip((image_ds, label_ds))
    dataset = dataset.shuffle(buffer_size=len(image_paths), seed=seed) if shuffle else dataset
    dataset.class_names =class_names

    return dataset


def landmark_dataset_generator(
        directory: os.PathLike,
        batch_size: int = 32,
        # image_size: tuple[int, int] = (256, 256),
        shuffle: bool = True,
        seed: int = 42,
        label_mode: str = 'int',
        # validation_split: float = None,
        # augmentation parameters
        augmenter: tf.keras.Sequential = None,
):
    # IMAGE DATASET
    # image_ds = tf.keras.utils.image_dataset_from_directory(
    #     directory,
    #     batch_size=None,
    #     image_size=image_size,
    #     shuffle=shuffle,
    #     seed=seed,
    #     label_mode=label_mode,
    #     validation_split=validation_split,
    #     subset="both" if validation_split is not None else None
    # )

    image_ds = image_dataset_from_directory(directory, shuffle=shuffle, seed=seed)

    # get the index of the unknown class
    # if validation_split is not None:
    #     ds_ref = image_ds[0]
    # else:
    #     ds_ref = image_ds
    
    if "none" not in image_ds.class_names:
        image_ds.class_names.append("none")

    unknown_class_index = image_ds.class_names.index("none")
    
    # LANDMARK EXTRACTION FUNCTION
    @tf.py_function(Tout=[tf.float32, tf.int32])
    def get_landmark(image, label):
        landmarks_list = []
        with mp_hands.Hands(model_complexity=0,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
    
            result = hands.process(image.numpy().astype(np.uint8))

        try:
            landmarks_list.append(result.multi_handedness[0].classification[0].index)
        except (IndexError, TypeError):
            # undetectable hand
            tf.logging.info("Hand not detected")
            return np.zeros(127, dtype=np.float32), unknown_class_index

        hl_wl = list(result.multi_hand_landmarks[0].landmark) + list(result.multi_hand_world_landmarks[0].landmark)
        # for landmark in result.hand_landmarks[0] + result.hand_world_landmarks[0]:
        for landmark in hl_wl:
            landmarks_list.extend([landmark.x, landmark.y, landmark.z])
        
        landmarks_array = np.array(landmarks_list, dtype=np.float32)
        return landmarks_array, label

    # LANDMARK EXTRACTION PIPELINE
    def img_to_landmark(image, label):
        landmarks, label = get_landmark(image, label)
        landmarks.set_shape((127,))
        label.set_shape(())
        return landmarks, label

    # DATASET PIPELINE
    # if validation_split is not None:
    #     train_ds = image_ds[0]
    #     val_ds = image_ds[1]

    #     # augment
    #     if augmenter is not None:
    #         train_ds = train_ds.map(lambda x, y: (augmenter(x), y))

    #     # extract landmarks
    #     train_ds = train_ds.map(img_to_landmark)
    #     val_ds = val_ds.map(img_to_landmark)

    #     # batch and prefetch
    #     train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    #     val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    #     # set class names
    #     train_ds.class_names = image_ds[0].class_names
    #     val_ds.class_names = image_ds[0].class_names

    #     return train_ds , val_ds
    
    # else:
    if augmenter is not None:
        augmented_ds = image_ds.map(lambda x, y: (augmenter(x), y))
    else:
        augmented_ds = image_ds

    landmark_ds = augmented_ds.map(img_to_landmark)

    # batch and prefetch
    landmark_ds = landmark_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # set class names
    landmark_ds.class_names = image_ds.class_names

    return landmark_ds