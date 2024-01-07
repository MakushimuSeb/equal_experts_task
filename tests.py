import unittest
import requests
import yaml
import pandas as pd
from unittest.mock import patch

import api
import train_model
import predict


class TestTrainModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config.yaml", "r") as file:
            cls.config = yaml.safe_load(file)

    def test_read_dataset(self):
        """
        Checks if the dataset loaded is an instance of pandas.DataFrame and
        if the dataframe has the expected number of columns (9)
        """
        df = train_model.read_dataset(self.config)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df.columns), 9)

    def test_train_model(self):
        """
        Loads the dataset, prepares the training data, and trains the model.
        The test then checks if the returned model is not None
        and whether the model has the fit attribute,
        indicating it's a trained model.
        """
        df = train_model.read_dataset(self.config)
        X_train, _, Y_train, _ = train_model.prepare_training_data(
            df,
            self.config
            )
        model = train_model.train_model(X_train, Y_train, self.config)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, "fit"))

    def test_evaluate_model(self):
        """
        Prepares the data, trains the model, and evaluates it.
        The test checks if the returned score is a float.
        """
        df = train_model.read_dataset(self.config)
        X_train, X_test, Y_train, Y_test = train_model.prepare_training_data(
            df,
            self.config
            )
        model = train_model.train_model(X_train, Y_train, self.config)
        score = train_model.evaluate_model(model, X_test, Y_test)
        self.assertIsInstance(score, float)


class TestAPI(unittest.TestCase):
    def setUp(self):
        api.app.testing = True
        self.client = api.app.test_client()

    def test_predict_endpoint(self):
        """
        Sends a POST request to the /predict endpoint
        and checks if the response status code is 200.
        """
        response = self.client.post("/predict", json=[1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(response.status_code, 200)

    @patch('requests.get')
    def test_model_loading(self, mock_get):
        """
        Checks if the mocked response matches
        the expected JSON object with a prediction.
        """
        mock_get.return_value.json.return_value = {"prediction": 1}
        response = requests.get("http://localhost:5000/predict")
        self.assertEqual(response.json(), {"prediction": 1})


class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("config.yaml", "r") as file:
            cls.config = yaml.safe_load(file)

    @patch('requests.post')
    def test_integration_predict(self, mock_post):
        """
        Checks if the result contains the "prediction" key
        and if the value of this key matches the expected prediction.
        """
        mock_post.return_value.json.return_value = {"prediction": [1]}
        result = predict.predict(self.config, mode="local_url")
        self.assertIn("prediction", result)
        self.assertEqual(result["prediction"], [1])


if __name__ == '__main__':
    unittest.main()
