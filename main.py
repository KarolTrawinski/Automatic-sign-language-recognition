import os

from class_handler import PredictOperator, SavingOperaptor
from learning_AI import create_model
from value_storage import Data


def print_menu() -> str:
    menu = f"""
    ################################################
    Select training setup:
    1. Select sign to save
    2. Learn images by AI
    3. Turn on camera and predict sign!
    ################################################
    """
    return menu


def print_saving_menu() -> str:
    menu = """ 
    ################################################
        Insert sign to save:"""
    return menu


def print_predict_menu() -> str:
    string = """ 
    ################################################
        Select tenserflow model:
        1. Google model
        2. Locally learned model
    ################################################
    """
    return string


def check_tenserflow_model(model_path):
    if os.path.exists(model_path):
        return True
    else:
        print("Model file didn't exist!")
        exit()


def select_predict_model():
    inp = input(print_predict_menu())
    if inp == "1":
        model_path = Data.MODEL_PATH.value[0]
        labels_path = Data.LABELS_PATH.value[0]
    elif inp == "2":
        model_path = Data.MODEL_PATH.value[1]
        labels_path = Data.LABELS_PATH.value[1]
        check_labels(labels_path)
    else:
        print("Wrong model!")
        exit()
    check_tenserflow_model(model_path)
    return model_path, labels_path


def check_labels(labels) -> None:
    signs = os.listdir(Data.LEARNING_FOLDER_PATH.value)
    id = 0
    with open(labels, "w") as f:
        for s in signs:
            f.write(f"{id} {s}\n")
            id += 1


if __name__ == "__main__":
    inp = input(print_menu())
    if inp == "1":
        sign = input(print_saving_menu()).upper()
        print("Click 's' to save image!")
        learning_path = f"{Data.LEARNING_FOLDER_PATH.value}/{sign}"
        saving_operator = SavingOperaptor(learning_path)
        saving_operator.handle_saving()
    elif inp == "2":
        create_model()
    elif inp == "3":
        model_path, labels_path = select_predict_model()
        predict_operator = PredictOperator(model_path, labels_path)
        predict_operator.predict()
