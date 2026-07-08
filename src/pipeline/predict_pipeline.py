import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("data", "model.pkl")
            preprocessor_path = os.path.join("data", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        age: int,
        sex: str,
        chest_pain_type: str,
        resting_bp: int,
        cholesterol: int,
        fasting_bs: int,
        max_hr: int,
        resting_ecg: str,
        exercise_angina: str,
        oldpeak: float,
        st_slope: str,
    ):
        self.age = age
        self.sex = sex
        self.chest_pain_type = chest_pain_type
        self.resting_bp = resting_bp
        self.cholesterol = cholesterol
        self.fasting_bs = fasting_bs
        self.max_hr = max_hr
        self.resting_ecg = resting_ecg
        self.exercise_angina = exercise_angina
        self.oldpeak = oldpeak
        self.st_slope = st_slope

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Age": [self.age],
                "Sex": [self.sex],
                "ChestPainType": [self.chest_pain_type],
                "RestingBP": [self.resting_bp],
                "Cholesterol": [self.cholesterol],
                "FastingBS": [self.fasting_bs],
                "MaxHR": [self.max_hr],
                "RestingECG": [self.resting_ecg],
                "ExerciseAngina": [self.exercise_angina],
                "Oldpeak": [self.oldpeak],
                "ST_Slope": [self.st_slope],
            }
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
