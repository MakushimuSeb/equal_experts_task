# Task Descrition Excerpts and Assumptions

*Our client has many data scientists creating machine learning models from the data. The
models are created using python and they are deployed in an ad-hoc fashion. We have been
asked to create an ML pipeline to support CI/CD of ML development. For a cloud
environment of your choice create a simple infrastructure to*
- *Store a machine learning model*
- *Run Tests against it*
- *Promote to production*
- *Your approach should support versioning of the model.*

*Things we value:*
- *Code organisation – The code must speak for itself.*
- *Simplicity – We value simplicity, solutions should reflect the difficulty of the assigned task, and should NOT be overly complex. You should be able to explain your methodology choices.*
- *Self-explanatory code – For instance, variables and methods should have good clean names. The code should be simple and straightforward to understand.*


Based on the task description, I understand that the solution must include the following components:
- Storage and versioning of ML models
- Tests
- Mechanics to promote the model to production
- The solution shall be implemented in cloud infrastructure
- The solution shall be simple by design (which I understand as straightforward architecture, minimum dependencies, minimum external tools and minimum lines of code)


While designing the solution I proceeded from the following assuptions:
- The solution shall process one event at a time 
- There shall be a cloud API endpoint evailable to call the model and get the prediction
- The amount of requests shall not exceed hundreds per minute (the solution does not require a load balancer and there is no need to handle calls asyncroniously)
- Though the pipeline shall be implemented using cloud infrastructure, it shall be as simple as possible and can be easily migrated from one cloud provier to another

# ML Pipeline


### Launching prediction service locally:

```bash
docker build -t prediction_app .
docker run -p 5000:5000 prediction_app
```

The local API is available at the endpoint ```http://localhost:5000/predict```

A call example ```curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '[6, 148, 72, 35, 0, 33.6, 0.627, 50]'```

The cloud API is available at the endpoint ```http://ec2-54-209-225-200.compute-1.amazonaws.com:5000/predict```

A call example ```curl -X POST http://ec2-54-209-225-200.compute-1.amazonaws.com:5000/predict -H "Content-Type: application/json" -d '[6, 148, 72, 35, 0, 33.6, 0.627, 50]'```


## Model training and code development

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Model training

- Change parameters in ```config.yaml```
- ```python train_model.py```
- Trained model in ```.joblib``` format will be saved locally in ```/models``` and pushed to ```s3://equalexpertstask/models/``` (for simplicity and only for the purpose of this exersice the bucket is publicly accessible for the limited time)

Model versioning is implemented by assigning the date and time of the .joblib file's creation to the model name.

### Automated prediction service

The service picks random row from the dataset and sends the request to API

```bash
python predict.py --local to call local API
python prredict.py --remote to call cloud API
```

### Running tests

```python -m unittest```

### Deploying to production

On every ```push``` to Github, Github Actions are triggered and 
- check that code corresponds to PEP8 requirements
- run tests
- SSHing into EC2, pulling the latest version of code
- Building and running Docker container with the latest model available on S3

