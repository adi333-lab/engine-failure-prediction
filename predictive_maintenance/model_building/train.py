import os
import pandas as pd
import numpy as np
import joblib
import mlflow

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import classification_report, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import xgboost as xgb

# MLflow setup
mlflow.set_tracking_uri("https://brickred-deidre-unbelievingly.ngrok-free.dev")
mlflow.set_experiment("Engine_Failure_MLops")


#  Load preprocessed data from Hugging Face dataset repository
DATASET_REPO = "adi333/engine-failure-prediction"

Xtrain = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/ytest.csv").squeeze()

print(" Data loaded from Hugging Face dataset repo.")

numeric_features = Xtrain.columns.tolist()

# Data Clipping (from your successful code)
lower_quantile = Xtrain[numeric_features].quantile(0.01)
upper_quantile = Xtrain[numeric_features].quantile(0.99)

Xtrain[numeric_features] = Xtrain[numeric_features].clip(
    lower=lower_quantile,
    upper=upper_quantile,
    axis=1
)
Xtest[numeric_features] = Xtest[numeric_features].clip(
    lower=lower_quantile,
    upper=upper_quantile,
    axis=1
)

# Creating validation dataset for early stopping
X_train_main, X_val, y_train_main, y_val = train_test_split(
    Xtrain, ytrain, test_size=0.2, random_state=42
)

# Compute class imbalance weight
class_weight = y_train_main.value_counts()[0] / y_train_main.value_counts()[1]

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_features)
    ]
)

# Processing validation dataset 
preprocessor.fit(X_train_main)
X_val_processed = preprocessor.transform(X_val)

# Model Definition
xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    n_estimators=1000,
    early_stopping_rounds=50 
)

#  Pipeline (from your successful code)
model_pipeline = make_pipeline(preprocessor, xgb_model)

#Hyperparameters for fine-tuning
param_dist = {
    'xgbclassifier__max_depth': [3, 4, 5, 6],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6, 0.7],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6, 0.7],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [5.0, 10.0, 20.0, 50.0],
    'xgbclassifier__gamma': [0.5, 1.0, 2.0, 5.0],
    'xgbclassifier__subsample': [0.6, 0.7, 0.8]
}

# Start MLflow run
with mlflow.start_run():
    # Search Strategy
    random_search = RandomizedSearchCV(
        model_pipeline,
        param_distributions=param_dist,
        n_iter=50,
        cv=5, # This will split X_train_main
        scoring='f1',
        verbose=1,
        n_jobs=-1
    )
    
    # Fitting the model (from your successful code)
    random_search.fit(
        X_train_main,  # <-- Use the smaller training set
        y_train_main,
        # Pass the eval_set to the 'xgbclassifier' step
        xgbclassifier__eval_set=[(X_val_processed, y_val)], 
        xgbclassifier__verbose=False
    )

    # Log hyperparameters
    mlflow.log_params(random_search.best_params_)

    # Store the best model
    best_model = random_search.best_estimator_

    # Optimal Threshold 
    y_scores = best_model.predict_proba(Xtest)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(ytest, y_scores)

    # Find best threshold by maximizing F1-score
    f1_scores = (2 * precisions * recalls) / (precisions + recalls + 1e-9)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_threshold = thresholds[best_f1_index]

    print(f" Optimal threshold found: {best_threshold:.4f} (Best Test F1: {best_f1:.4f})")
    mlflow.log_metric("best_test_f1_score", best_f1)
    mlflow.log_metric("optimal_threshold", best_threshold)

    # Use the BEST threshold for final predictions
    classification_threshold = best_threshold 

    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report.get('accuracy', 0),
        "train_precision": train_report.get('1', {}).get('precision', 0),
        "train_recall": train_report.get('1', {}).get('recall', 0),
        "train_f1-score": train_report.get('1', {}).get('f1-score', 0),
        "test_accuracy": test_report.get('accuracy', 0),
        "test_precision": test_report.get('1', {}).get('precision', 0),
        "test_recall": test_report.get('1', {}).get('recall', 0),
        "test_f1-score": test_report.get('1', {}).get('f1-score', 0)
    })

    # Save the model locally
    model_path = "best_engine_failure_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    api = HfApi(token=os.getenv("HF_TOKEN"))
    repo_id = "adi333/engine-failure-prediction-model"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # Upload the model
    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="best_engine_failure_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f" Model uploaded to Hugging Face model repo: {repo_id}")
print("\n🚀 Training + Logging + Upload COMPLETE!")
