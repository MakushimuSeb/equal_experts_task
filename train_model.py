import os
from datetime import datetime

import joblib
import pandas as pd
import yaml
import boto3
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def read_dataset(config):
    path = config["dataset"]["path"]
    names = config["dataset"]["names"]
    dataframe = pd.read_csv(path, names=names)
    return dataframe


def prepare_training_data(dataframe, config):
    X = dataframe.iloc[:, :-1].values
    Y = dataframe.iloc[:, -1].values

    test_size = config["model"]["test_size"]
    seed = config["model"]["seed"]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    return X_train, X_test, Y_train, Y_test


def train_model(X_train, Y_train, config):
    max_iter = config["model"]["max_iterations"]
    model = LogisticRegression(max_iter=max_iter)
    model.fit(X_train, Y_train)
    return model


def evaluate_model(model, X_test, Y_test):
    return model.score(X_test, Y_test)


def save_model_to_s3(model, config):

    model_dir = config["model"]["model_dir"]
    model_name = config["model"]["model_name"]

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    filename = f"{model_name}_v{timestamp}.joblib"
    local_file_path = os.path.join(model_dir, filename)

    joblib.dump(model, local_file_path)

    s3_bucket = config['aws']['s3_bucket']
    s3_path = f'models/{filename}'

    s3_client = boto3.client("s3")

    try:
        s3_client.upload_file(local_file_path, s3_bucket, s3_path)
        print(f"Model uploaded to s3://{s3_bucket}/{s3_path}")
    except Exception as e:
        print(f"Error uploading model to S3: {e}")


if __name__ == "__main__":
    print("Reading dataset...")
    dataframe = read_dataset(config)
    print("Preparing training data...")
    X_train, X_test, Y_train, Y_test = prepare_training_data(dataframe, config)
    print("Training model...")
    model = train_model(X_train, Y_train, config)
    result = evaluate_model(model, X_test, Y_test)
    print("Model is trained. Accuracy: %.3f%%" % (result * 100.0))
    save_model_to_s3(model, config)
