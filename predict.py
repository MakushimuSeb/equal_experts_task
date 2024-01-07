import argparse
import requests
import yaml

from train_model import read_dataset

parser = argparse.ArgumentParser(description='Run the prediction model.')
parser.add_argument('--local', action='store_true', help='Use local API')
parser.add_argument('--remote', action='store_true', help='Use remote API')
args = parser.parse_args()

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


def predict(config, mode):
    api_url = config["api"][mode]
    dataframe = read_dataset(config)
    random_row = dataframe.sample().values[0][:-1].astype(float)
    features_list = random_row.tolist()
    headers = {"Content-Type": "application/json"}
    response = requests.post(api_url, json=features_list, headers=headers)
    print(f"API URL: {api_url}")
    print(f"FEATURES: {features_list}")
    return response.json()


if __name__ == "__main__":
    if args.local:
        mode = "local_url"
        print("RUNNING IN LOCAL MODE.")
    elif args.remote:
        mode = "remote_url"
        print("RUNNING IN REMOTE MODE.")
    else:
        raise ValueError("Please specify either --local or --remote.")

    result = predict(config, mode)
    print(result)
