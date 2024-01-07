import joblib
import numpy as np
import yaml
import boto3
from botocore import UNSIGNED
from botocore.client import Config
from flask import Flask, jsonify, request


with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

app = Flask(__name__)


def download_latest_model_from_s3(bucket, model_dir):
    s3_client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=model_dir)

    if 'Contents' not in response:
        raise FileNotFoundError("No model files found in the S3 bucket.")

    # Sorting the returned objects by last modified date
    all_models = response['Contents']
    latest_model = max(all_models, key=lambda x: x['LastModified'])

    # Download the latest model file
    model_file = latest_model['Key']
    model_file_name = model_file.split('/')[-1]
    model_dir = config['model']['model_dir']

    local_model_path = f'{model_dir}/{model_file_name}'

    s3_client.download_file(bucket, model_file, local_model_path)
    return local_model_path


model_path = download_latest_model_from_s3(
    config['aws']['s3_bucket'],
    config['model']['model_dir']
    )

model = joblib.load(model_path)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    input = np.array(data).reshape(1, -1)
    prediction = model.predict(input)
    return jsonify({"prediction": int(prediction)})


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
