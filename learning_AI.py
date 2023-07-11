import os

import tensorflow as tf

from value_storage import Data


class Model:
    def __init__(self) -> None:
        self.path = Data.LEARNING_FOLDER_PATH.value
        self.model_path = Data.LABELS_PATH.value[1]
        self.folder_size = 0
        self.labels = []
        self.train_ds = None
        self.val_ds = None

    def set_folder_size_and_labels(self) -> None:
        for _, dirs, files in os.walk(self.path):
            self.folder_size = len(dirs)
            self.labels = dirs
            break

    def setup_ds(self) -> None:
        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            class_names=self.labels,
            image_size=(224, 224),
            batch_size=16,
            seed=123,
            validation_split=0.1,
            subset="training",
        )
        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.path,
            class_names=self.labels,
            image_size=(224, 224),
            batch_size=16,
            seed=123,
            validation_split=0.1,
            subset="validation",
        )

    def setup_model(self) -> None:
        model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Conv2D(
                    32, self.folder_size, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(
                    32, self.folder_size, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Conv2D(
                    32, self.folder_size, activation="relu"),
                tf.keras.layers.MaxPooling2D(),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(self.folder_size),
            ]
        )

        model.compile(
            optimizer="adam",
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=["accuracy"],
        )

        model.fit(self.train_ds, validation_data=self.val_ds, epochs=2)
        model.save(self.model_path)


def create_model() -> None:
    model = Model()
    model.set_folder_size_and_labels()
    model.setup_ds()
    model.setup_model()
    print("New model created!")
