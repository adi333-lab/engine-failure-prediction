import pandas as pd
import os

# sklearn preprocessing
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Hugging Face
from huggingface_hub import HfApi

# Initialize HF API with token (must be stored as environment variable)
api = HfApi(token=os.getenv("HF_TOKEN"))

# Load Dataset from HF Hub
DATASET_PATH = "hf://datasets/adi333/engine-failure-prediction/data/engine_data.csv"

df = pd.read_csv(DATASET_PATH)
print(" Dataset loaded successfully.")

# Define Features & Target

target_col = "Engine Condition"   # label
X = df.drop(columns=[target_col])
y = df[target_col]

# Identify numeric columns
numeric_features = X.columns.tolist()

# Preprocessing Pipeline

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # handle missing values
    ("scaler", StandardScaler())                     # normalize sensor values
])

preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", numeric_pipeline, numeric_features)
    ]
)

# Apply preprocessing
X_preprocessed = preprocessor.fit_transform(X)
print(" Preprocessing pipeline applied successfully.")

# Convert back to DataFrame to save
X_preprocessed = pd.DataFrame(X_preprocessed, columns=numeric_features)

# Train-Test Split

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_preprocessed, y, test_size=0.2, random_state=42
)

print(" Dataset split into train & test.")

# Save Locally

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

print(" Data splits saved locally.")

# Upload Files to Hugging Face Dataset Repo

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path,
        repo_id="adi333/engine-failure-prediction",
        repo_type="dataset",
    )
    print(f" Uploaded {file_path} to Hugging Face dataset repo.")

print("\n Preprocessing + Split + Upload COMPLETE!")
