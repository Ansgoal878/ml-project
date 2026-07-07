import os
import sys

import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_classification_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        best_params_report = {}

        for model_name, model in models.items():
            para = param[model_name]

            gs = GridSearchCV(model, para, cv=3, scoring="f1")
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)

            if hasattr(model, "predict_proba"):
                y_score = model.predict_proba(X_test)[:, 1]
            else:
                y_score = model.decision_function(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_score)

            report[model_name] = {
                "accuracy_score": acc,
                "f1_score": f1,
                "roc_auc_score": roc_auc,
            }
            best_params_report[model_name] = gs.best_params_

            logging.info(
                f"{model_name} -> accuracy: {acc:.4f}, f1: {f1:.4f}, roc_auc: {roc_auc:.4f}"
            )

        return report, best_params_report

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)