import os
import sys
from dataclasses import dataclass
 
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
 
from src.exception import CustomException
from src.logger import logging
 
from src.utils import save_object, evaluate_classification_models
 
 
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("data", "model.pkl")
 
 
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
 
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
 
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(),
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "XGBClassifier": XGBClassifier(),
                "CatBoosting Classifier": CatBoostClassifier(verbose=False),
                "AdaBoost Classifier": AdaBoostClassifier(),
            }
 
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy", "log_loss"],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest": {
                    # 'criterion':['gini','entropy','log_loss'],
                    # 'max_features':['sqrt','log2',None],
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    # 'criterion':['friedman_mse','squared_error'],
                    # 'max_features':['sqrt','log2'],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Logistic Regression": {
                    "C": [0.01, 0.1, 1, 10, 100],
                },
                "XGBClassifier": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Classifier": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }
 
            model_report, best_params_report = evaluate_classification_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )
 
            results_df = pd.DataFrame([
                {
                    "model": name,
                    "hyperparameters": str(best_params_report[name]),
                    "accuracy_score": scores["accuracy_score"],
                    "f1_score": scores["f1_score"],
                    "roc_auc_score": scores["roc_auc_score"],
                }
                for name, scores in model_report.items()
            ]).sort_values("roc_auc_score", ascending=False)
 
            results_path = os.path.join("data", "model_results.csv")
            results_df.to_csv(results_path, index=False)
            logging.info(f"Model results saved to {results_path}")
 
            # Select the best model based on roc_auc_score
            # (roc_auc is generally preferred for medical/imbalanced classification tasks)
            best_model_name = max(
                model_report, key=lambda name: model_report[name]["roc_auc_score"]
            )
            best_model_scores = model_report[best_model_name]
            best_model = models[best_model_name]
 
            if best_model_scores["roc_auc_score"] < 0.6:
                raise CustomException("No best model found")
 
            logging.info(
                f"Best model found: {best_model_name} "
                f"(accuracy: {best_model_scores['accuracy_score']:.4f}, "
                f"f1: {best_model_scores['f1_score']:.4f}, "
                f"roc_auc: {best_model_scores['roc_auc_score']:.4f})"
            )
 
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )
 
            y_pred = best_model.predict(X_test)
            if hasattr(best_model, "predict_proba"):
                y_score = best_model.predict_proba(X_test)[:, 1]
            else:
                y_score = best_model.decision_function(X_test)
 
            final_accuracy = accuracy_score(y_test, y_pred)
            final_f1 = f1_score(y_test, y_pred)
            final_roc_auc = roc_auc_score(y_test, y_score)
 
            return {
                "accuracy_score": final_accuracy,
                "f1_score": final_f1,
                "roc_auc_score": final_roc_auc,
            }
 
        except Exception as e:
            raise CustomException(e, sys)