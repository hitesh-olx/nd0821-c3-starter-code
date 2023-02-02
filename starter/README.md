# Deploying a Machine Learning Model on Heroku with FastAPI  

This project contains the development of a classification model on Census Bureau data. 
The main goal is to robustly deploy a machine learning model into production.  
This includes: 
* versioning the data and the models using Data Version Control (DVC).
* testing the code using pytest
* deploying the model using the FastAPI package and creating API tests on Heroku
* incorporating the ML pipeline into a CI/CD framework using GitHub Actions.

### Environment Set up  

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment

    ```
    conda env create -f environment.yml
    ```
    * activate the env
    ```
    conda activate heroku
    ````


### Model  

* To train the model run:
``` 
python src/train_model.py
```

* or run the entire ML pipeline which starts a local server where you can test the model
```
python main.py
```

### Heroku deployment  

* Alternatively test the model live on Heroku by executing a POST request:

```
python heroku_api_request.py
```

### GitHub Actions  

The machine learning pipeline is deployed automatically in a CI/CD fashion. The model artifacts and the data are pulled
from a S3 bucket using DVC. After successfully passing the tests, the code is automaticaly pushed to the Heroku instance.