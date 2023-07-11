import fnmatch
import os
import time

import cv2
import numpy as np
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

from value_storage import Data


class Operator:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)
        self.detector = HandDetector(maxHands=1)
        self.space = Data.SPACE.value
        self.img_size = Data.IMG_SIZE.value

    def resize_img(self, img, w, h):
        background = np.ones((self.img_size, self.img_size, 3), np.uint8) * 255
        ratio = h / w
        try:
            if ratio > 1:
                scale = self.img_size / h
                width = int(w * scale)
                img_resized = cv2.resize(img, (width, self.img_size))
                start = int((self.img_size - width) / 2)
                background[:, start: width + start] = img_resized

            else:
                scale = self.img_size / w
                height = int(h * scale)
                img_resized = cv2.resize(img, (self.img_size, height))
                start = int((self.img_size - height) / 2)
                background[start: start + height, :] = img_resized
            return background
        except Exception:
            pass

    def return_data(self):
        while True:
            success, img = self.cap.read()
            hand, img = self.detector.findHands(img)
            if hand:
                hand = hand[0]
                x, y, w, h = hand["bbox"]
                handCrop = img[
                    y - self.space: y + h + self.space,
                    x - self.space: x + w + self.space,
                ]
                return {
                    "handCrop": handCrop,
                    "img": img,
                    "w": w,
                    "h": h,
                    "x": x,
                    "y": y,
                }


class SavingOperaptor(Operator):
    def __init__(self, path) -> None:
        super().__init__()
        self.path = path

    def handle_saving(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        while True:
            data = self.return_data()
            hand_img = self.resize_img(data["handCrop"], data["w"], data["h"])
            cv2.imshow("Image", data["img"])
            key = cv2.waitKey(1)
            if key == ord("s"):
                count = len(fnmatch.filter(os.listdir(self.path), "*.*"))
                print("File Count:", count)
                try:
                    cv2.imwrite(f"{self.path}/img_{count}.jpg", hand_img)
                except Exception:
                    pass


class PredictOperator(Operator):
    def __init__(self, model_path, labels_path) -> None:  # !New
        super().__init__()
        self.predicted_signs_list = None
        self.predicted_percentages_list = None
        self.model_path = model_path
        self.labels_path = labels_path
        self.classifier = None
        self.signs = None

    def setup_classifier(self) -> None:
        try:
            classifier = Classifier(self.model_path, self.labels_path)
            self.classifier = classifier
        except Exception:
            print("No model!!!!")

    def get_learned_signs(self) -> None:
        with open(self.labels_path) as f:
            signs_list = {}
            for line in f:
                id, sign = line.split()
                if id not in signs_list:
                    signs_list[id] = ""
                signs_list[id] += sign

        signs_list = [signs_list[key] for key in sorted(signs_list)]
        self.signs = signs_list
        self.predicted_signs_list = [0 for i in range(len(self.signs))]
        self.predicted_percentages_list = [0 for i in range(len(self.signs))]

    def resize_img(self, img, w, h) -> int:  # ! New
        background = super().resize_img(img, w, h)
        p, i = self.classifier.getPrediction(background)
        return p, i

    def add_to_test_list(self, p, id):  # !New
        try:
            self.predicted_signs_list[id] += 1
            self.predicted_percentages_list = [self.predicted_percentages_list[i] + p[i]
                                               for i in range(len(self.predicted_percentages_list))]
        except Exception as e:
            print(e)

        print(f'len(p) = {len(p)}, id = {id} p= {p}')

    def predict(self) -> None:  # ! New
        self.get_learned_signs()
        self.setup_classifier()
        while True:
            time.sleep(0.001)
            try:
                data = self.return_data()
                p, id = self.resize_img(
                    data["handCrop"], data["w"], data["h"])
                self.add_to_test_list(p, id)
                formated_p = str("{:.1f}%".format(p[id]*100))
                formated_text = f'{self.signs[id]} {formated_p}'
                cv2.putText(
                    data["img"],
                    formated_text,
                    (data["x"]-40, data["y"]-60),
                    cv2.FONT_HERSHEY_COMPLEX,
                    2,
                    (0, 0, 255),
                )
                print(
                    f'self.predicted_signs_list = {self.predicted_signs_list}, self.predicted_percentages_list= {self.predicted_percentages_list}')
                cv2.imshow("Image", data["img"])
                cv2.waitKey(1)
            except Exception as e:
                print(e)
                pass
