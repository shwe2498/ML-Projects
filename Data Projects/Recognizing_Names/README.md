# Recognizing Person Names Using NLP and Machine Learning

## Introduction
This project aims to develop a classifier to recognize valid person names from a given string using Natural Language Processing (NLP) and machine learning techniques.

## Data Sources
- **Person Names**: Retrieved from [DBPedia](http://dbpedia.org/sparql) using SPARQL query to fetch names classified under `dbo:Person`.
- **Common English Words**: Downloaded from a zipped archive containing a JSON file of words.

## Libraries Used
- `SPARQLWrapper`: To fetch data from DBPedia.
- `zipfile` and `json`: To read and process the common words data.
- `CountVectorizer`: For feature extraction from text.
- `Word2Vec`: For generating word embeddings.
- `StandardScaler`: For feature scaling.
- `RandomForestClassifier`, `GridSearchCV`: For model training and hyperparameter tuning.
- `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`: For model evaluation.

## Feature Engineering
- **Character N-grams**: Extracted using `CountVectorizer`.
- **Word Embeddings**: Generated using `Word2Vec`.
- **Length and Capital Letter Features**: Length of the string and the number of capital letters in the string.

## Model Training
- **Random Forest Classifier**: Used to classify names.
- **Hyperparameter Tuning**: Performed using `GridSearchCV` to find the best model parameters.

## Model Evaluation
Evaluated the model using precision, recall, F1 score, and ROC AUC score.

## Results
- **Precision**: 
- **Recall**: 
- **F1 Score**: 
- **ROC AUC Score**: 

## Usage
To run this project, install the required libraries and run the provided code.

```bash
pip install -r requirements.txt
python recognize_names.py
