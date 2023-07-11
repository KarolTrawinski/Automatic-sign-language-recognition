from enum import Enum


class Data(Enum):
    MODEL_PATH = [
        "google_model/keras_model.h5",
        "custom_model/custom_training_model.h5"
    ]
    LABELS_PATH = ["google_model/labels.txt", "custom_model/labels.txt"]
    LEARNING_FOLDER_PATH = "LearningData"
    SPACE = 25
    IMG_SIZE = 200
