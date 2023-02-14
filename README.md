# Deploying Machine Learning Model on Render

This project contains the development of a classification model[XG Boost Model] on Census Bureau data. 
The main goal is to robustly deploy a machine learning model into production.  
This includes: 
* testing using pytest
* deploying the model using the FastAPI package and creating API tests on Render
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

### Render deployment  

* Alternatively test the model live on Render by executing a POST request:

```
python render_api_request.py
```


