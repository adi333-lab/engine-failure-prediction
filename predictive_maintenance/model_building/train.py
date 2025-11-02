import os
import pandas as pd
import joblib
import mlflow

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

import xgboost as xgb


# MLflow setup
mlflow.set_tracking_uri("https://brickred-deidre-unbelievingly.ngrok-free.dev")
mlflow.set_experiment("Engine_Failure_MLops")


# ✅ Load preprocessed data from Hugging Face dataset repository
DATASET_REPO = "adi333/engine-failure-prediction"

Xtrain = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/Xtrain.csv")
Xtest  = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/Xtest.csv")
ytrain = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/ytrain.csv").squeeze()
ytest  = pd.read_csv(f"hf://datasets/adi333/engine-failure-prediction/ytest.csv").squeeze()

print(" Data loaded from Hugging Face dataset repo.")

numeric_features = Xtrain.columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("scaler", StandardScaler(), numeric_features)
    ]
)


# Compute class imbalance weight
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=class_weight,
    random_state=42,
    eval_metric='logloss'
)

# Pipeline
pipeline = Pipeline([
    ("scaler", preprocessor),
    ("model", xgb_model)
])


# Hyperparameters for XGBoost
param_grid = {
    'model__n_estimators': [50, 75, 100, 125, 150],
    'model__max_depth': [2, 3, 4],
    'model__colsample_bytree': [0.4, 0.5, 0.6],
    'model__colsample_bylevel': [0.4, 0.5, 0.6],
    'model__learning_rate': [0.01, 0.05, 0.1],
    'model__reg_lambda': [0.4, 0.5, 0.6],
}

# Start MLflow run
with mlflow.start_run():
    # Hyperparameter tuning with GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(Xtrain, ytrain)

    # Log hyperparameters
    mlflow.log_params(grid_search.best_params_)

    # Store the best model
    best_model = grid_search.best_estimator_

    # Set classification threshold
    classification_threshold = 0.45

    # Make predictions on the training and test data
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Evaluation
    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1-score": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1-score": test_report['1']['f1-score']
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

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_engine_failure_prediction_model_v1.joblib",
        path_in_repo="best_engine_failure_prediction_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )

print(f"✅ Model uploaded to Hugging Face model repo: {repo_id}")
print("\n🚀 Training + Logging + Upload COMPLETE!")
