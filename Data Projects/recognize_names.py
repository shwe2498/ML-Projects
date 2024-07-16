{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "gpuType": "V28"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    },
    "accelerator": "TPU"
  },
  "cells": [
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Introduction:\n",
        "\n",
        "This project aims to develop a classifier to recognize valid person names from a given string using Natural Language Processing (NLP) and machine learning techniques.\n",
        "\n"
      ],
      "metadata": {
        "id": "Ww46uM4MsyjT"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "!pip install SPARQLWrapper"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "hgxR7J3Ncmc0",
        "outputId": "380c610f-03ec-4dcd-ef3b-bdeea7ad2785"
      },
      "execution_count": 1,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Collecting SPARQLWrapper\n",
            "  Downloading SPARQLWrapper-2.0.0-py3-none-any.whl (28 kB)\n",
            "Collecting rdflib>=6.1.1 (from SPARQLWrapper)\n",
            "  Downloading rdflib-7.0.0-py3-none-any.whl (531 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m531.9/531.9 kB\u001b[0m \u001b[31m9.7 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hCollecting isodate<0.7.0,>=0.6.0 (from rdflib>=6.1.1->SPARQLWrapper)\n",
            "  Downloading isodate-0.6.1-py2.py3-none-any.whl (41 kB)\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m41.7/41.7 kB\u001b[0m \u001b[31m5.5 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25hRequirement already satisfied: pyparsing<4,>=2.1.0 in /usr/local/lib/python3.10/dist-packages (from rdflib>=6.1.1->SPARQLWrapper) (3.1.2)\n",
            "Requirement already satisfied: six in /usr/local/lib/python3.10/dist-packages (from isodate<0.7.0,>=0.6.0->rdflib>=6.1.1->SPARQLWrapper) (1.16.0)\n",
            "Installing collected packages: isodate, rdflib, SPARQLWrapper\n",
            "Successfully installed SPARQLWrapper-2.0.0 isodate-0.6.1 rdflib-7.0.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Libraries Used:\n",
        "\n",
        "1. SPARQLWrapper: To fetch data from DBPedia.\n",
        "2. zipfile and json: To read and process the common words data.\n",
        "3. CountVectorizer: For feature extraction from text.\n",
        "4. Word2Vec: For generating word embeddings.\n",
        "5. StandardScaler: For feature scaling.\n",
        "6. RandomForestClassifier, GridSearchCV: For model training and hyperparameter tuning.\n",
        "7. precision_score, recall_score, f1_score, roc_auc_score: For model evaluation.\n",
        "\n"
      ],
      "metadata": {
        "id": "5bFye77ntOUZ"
      }
    },
    {
      "cell_type": "code",
      "execution_count": 2,
      "metadata": {
        "id": "Fa8CLQu-_KQL"
      },
      "outputs": [],
      "source": [
        "from SPARQLWrapper import SPARQLWrapper, JSON\n",
        "import zipfile\n",
        "import json\n",
        "from sklearn.feature_extraction.text import CountVectorizer\n",
        "import numpy as np\n",
        "from gensim.models import Word2Vec\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split, GridSearchCV\n",
        "from sklearn.ensemble import RandomForestClassifier\n",
        "from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score"
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Data Sources:\n",
        "\n",
        "Person Names: Retrieved from DBPedia using SPARQL query to fetch names classified under dbo:Person.\n",
        "Common English Words: Downloaded from Kaggle (Dataset containing 479k English words) a zipped archive containing a JSON file of words.\n",
        "\n"
      ],
      "metadata": {
        "id": "aLX8kquCs6WE"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Iniitialize the SPARQL wrapper with DBPedia endpoint\n",
        "sparql = SPARQLWrapper(\"http://dbpedia.org/sparql\")\n",
        "\n",
        "# Set the SPARQL query\n",
        "sparql.setQuery(\"\"\"\n",
        "  PREFIX dbo: <http://dbpedia.org/ontology/>\n",
        "  PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n",
        "\n",
        "  SELECT ?name\n",
        "  WHERE {\n",
        "    ?person a dbo:Person .\n",
        "    ?person foaf:name ?name .\n",
        "  }\n",
        "  LIMIT 100000\n",
        "\"\"\")\n",
        "\n",
        "# Set the return format to JSON\n",
        "sparql.setReturnFormat(JSON)\n",
        "\n",
        "# Execute the query and convert the results to a DataFrame\n",
        "results = sparql.query().convert()\n",
        "\n",
        "# Extract the names from the results\n",
        "person_names = [result[\"name\"][\"value\"] for result in results[\"results\"][\"bindings\"]]\n",
        "\n",
        "# Print the first 10 names to verify\n",
        "print(person_names[:10])\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YdolC3NLc0BY",
        "outputId": "9ebfffcb-2e56-4cb9-d83e-5fc5db860893"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['CaMia Jackson', 'Cab Calloway', 'Cab Kaye', 'Cabbrini Foncette', 'Cabell Breckiniridge', 'Calvin Cabell Tennis', 'Cabeção', 'Luís Morais', 'Cabinho', 'Evanivaldo Castro Silva']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Download the Common English Words dataset\n",
        "with zipfile.ZipFile('archive.zip', 'r') as z:\n",
        "  with z.open('words_dictionary.json') as f:\n",
        "    common_words = json.load(f)\n",
        "\n",
        "# Convert the dictionary to a list of words\n",
        "common_words = list(common_words.keys())\n",
        "\n",
        "# Print the first 10 words to verify\n",
        "print(common_words[:10])"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "xXWm0K32gjB6",
        "outputId": "117c915e-d554-43c3-aefd-d335b85f67cf"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['a', 'aa', 'aaa', 'aah', 'aahed', 'aahing', 'aahs', 'aal', 'aalii', 'aaliis']\n"
          ]
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Feature Engineering:\n",
        "\n",
        "1. Character N-grams: Extracted using CountVectorizer.\n",
        "2. Word Embeddings: Generated using Word2Vec.\n",
        "3. Length and Capital Letter Features: Length of the string and the number of capital letters in the string.\n",
        "\n"
      ],
      "metadata": {
        "id": "inIiKU6ptfOq"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "#Sampling\n",
        "sample_fraction = 0.05\n",
        "person_names_sample = pd.Series(person_names).sample(frac=sample_fraction, random_state=42).tolist()\n",
        "common_words_sample = pd.Series(common_words).sample(frac=sample_fraction, random_state=42).tolist()"
      ],
      "metadata": {
        "id": "m7m16__wmXFT"
      },
      "execution_count": 6,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "#Feature Engineering\n",
        "vectorizer = CountVectorizer(analyzer='char', ngram_range=(2, 5))\n",
        "X_names = vectorizer.fit_transform(person_names_sample)\n",
        "X_words = vectorizer.transform(common_words_sample)"
      ],
      "metadata": {
        "id": "ANgL9461hUI1"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Word Embeddings\n",
        "all_texts = person_names_sample + common_words_sample\n",
        "w2v_model = Word2Vec([list(text) for text in all_texts], vector_size=100, window=5, min_count=1, workers=4)\n",
        "w2v_features = np.array([np.mean([w2v_model.wv[word] for word in text if word in w2v_model.wv] or [np.zeros(100)], axis=0) for text in all_texts])"
      ],
      "metadata": {
        "id": "6NCUZUBvhnDU"
      },
      "execution_count": 8,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Lenth and Capital Letter Feature\n",
        "lengths = np.array([len(text) for text in all_texts]).reshape(-1, 1)\n",
        "capitals = np.array([sum(1 for char in text if char.isupper()) for text in all_texts]).reshape(-1, 1)"
      ],
      "metadata": {
        "id": "BALhBjTLiNZ-"
      },
      "execution_count": 9,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Combine Features\n",
        "combined_features = np.hstack((vectorizer.transform(all_texts).toarray(), w2v_features, lengths, capitals))\n",
        "\n",
        "X = combined_features\n",
        "y = np.hstack((np.ones(len(person_names_sample)), np.zeros(len(common_words_sample))))"
      ],
      "metadata": {
        "id": "6QqgV_QEiZm9"
      },
      "execution_count": 10,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Scale Features\n",
        "scaler = StandardScaler()\n",
        "X = scaler.fit_transform(X)"
      ],
      "metadata": {
        "id": "o3WzFIsKjUo5"
      },
      "execution_count": 11,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Split data\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
      ],
      "metadata": {
        "id": "Pxn2XowSn4vO"
      },
      "execution_count": 12,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Add edge cases to training data\n",
        "edge_cases = [\"A\", \"B\", \"C\", \"@\", \"John-Doe\", \"jane_doe\", \"O'Reilly\"]\n",
        "edge_case_labels = [1, 1, 1, 0, 1, 1, 1]\n",
        "\n",
        "edge_case_features = np.hstack((\n",
        "    vectorizer.transform(edge_cases).toarray(),\n",
        "    np.array([np.mean([w2v_model.wv[word] for word in text if word in w2v_model.wv] or [np.zeros(100)], axis=0) for text in edge_cases]),\n",
        "    np.array([len(text) for text in edge_cases]).reshape(-1, 1),\n",
        "    np.array([sum(1 for char in text if char.isupper()) for text in edge_cases]).reshape(-1, 1)\n",
        "))\n",
        "\n",
        "X_train = np.vstack((X_train, edge_case_features))\n",
        "y_train = np.hstack((y_train, edge_case_labels))"
      ],
      "metadata": {
        "id": "wUKVuXlQo27I"
      },
      "execution_count": 13,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Model Training:\n",
        "\n",
        "Random Forest Classifier: Used to classify names.\n",
        "\n",
        "Hyperparameter Tuning: Performed using GridSearchCV to find the best model parameters.\n",
        "\n"
      ],
      "metadata": {
        "id": "qG8IxtcjtwjM"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize Random Forest model\n",
        "rf = RandomForestClassifier(random_state=42)"
      ],
      "metadata": {
        "id": "IGSXwv_oo_Ro"
      },
      "execution_count": 14,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Define parameter grid for Random Forest\n",
        "param_grid_rf = {\n",
        "    'n_estimators': [100, 200],\n",
        "    'max_depth': [None, 10, 20],\n",
        "    'min_samples_split': [2, 5],\n",
        "    'min_samples_leaf': [1, 2]\n",
        "}"
      ],
      "metadata": {
        "id": "kPkwJrcPo9cu"
      },
      "execution_count": 15,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Initialize GridSearchCV for Random Forest\n",
        "grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, scoring='f1', n_jobs=-1, verbose=2)\n",
        "grid_search_rf.fit(X_train, y_train)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 135
        },
        "id": "U7LmF_S9pEHK",
        "outputId": "a6e47ca3-101f-4367-a385-f3cdd0d904df"
      },
      "execution_count": 16,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Fitting 5 folds for each of 24 candidates, totalling 120 fits\n"
          ]
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": [
              "GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
              "             param_grid={'max_depth': [None, 10, 20],\n",
              "                         'min_samples_leaf': [1, 2],\n",
              "                         'min_samples_split': [2, 5],\n",
              "                         'n_estimators': [100, 200]},\n",
              "             scoring='f1', verbose=2)"
            ],
            "text/html": [
              "<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-1\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [None, 10, 20],\n",
              "                         &#x27;min_samples_leaf&#x27;: [1, 2],\n",
              "                         &#x27;min_samples_split&#x27;: [2, 5],\n",
              "                         &#x27;n_estimators&#x27;: [100, 200]},\n",
              "             scoring=&#x27;f1&#x27;, verbose=2)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-1\" type=\"checkbox\" ><label for=\"sk-estimator-id-1\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5, estimator=RandomForestClassifier(random_state=42), n_jobs=-1,\n",
              "             param_grid={&#x27;max_depth&#x27;: [None, 10, 20],\n",
              "                         &#x27;min_samples_leaf&#x27;: [1, 2],\n",
              "                         &#x27;min_samples_split&#x27;: [2, 5],\n",
              "                         &#x27;n_estimators&#x27;: [100, 200]},\n",
              "             scoring=&#x27;f1&#x27;, verbose=2)</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-2\" type=\"checkbox\" ><label for=\"sk-estimator-id-2\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-3\" type=\"checkbox\" ><label for=\"sk-estimator-id-3\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">RandomForestClassifier</label><div class=\"sk-toggleable__content\"><pre>RandomForestClassifier(random_state=42)</pre></div></div></div></div></div></div></div></div></div></div>"
            ]
          },
          "metadata": {},
          "execution_count": 16
        }
      ]
    },
    {
      "cell_type": "markdown",
      "source": [
        "\n",
        "\n",
        "> Model Evaluation:\n",
        "\n",
        "Evaluated the model using precision, recall, F1 score, and ROC AUC score.\n",
        "\n"
      ],
      "metadata": {
        "id": "v3uHXb6Zt25m"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "# Evaluate the model\n",
        "best_rf = grid_search_rf.best_estimator_\n",
        "y_pred = best_rf.predict(X_test)\n",
        "\n",
        "precision = precision_score(y_test, y_pred)\n",
        "recall = recall_score(y_test, y_pred)\n",
        "f1 = f1_score(y_test, y_pred)\n",
        "roc_auc = roc_auc_score(y_test, best_rf.predict_proba(X_test)[:, 1])\n",
        "\n",
        "print(f'Precision: {precision:.2f}')\n",
        "print(f'Recall: {recall:.2f}')\n",
        "print(f'F1 Score: {f1:.2f}')\n",
        "print(f'ROC AUC Score: {roc_auc:.2f}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "l6C_qbn9pHA8",
        "outputId": "1919a23d-9330-41cf-ee0a-fd2d2de588a9"
      },
      "execution_count": 17,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Precision: 1.00\n",
            "Recall: 0.99\n",
            "F1 Score: 1.00\n",
            "ROC AUC Score: 1.00\n"
          ]
        }
      ]
    }
  ]
}