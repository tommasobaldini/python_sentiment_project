from src import config
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys
sys.path.append(os.path.abspath('..'))  # Adds the parent directory to sys.path

import logging
# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_data():
    """Loads data from the SQLite database."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    query = f"SELECT cleaned_text, sentiment FROM {config.PROCESSED_TABLE}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def train_model(grid_search=False):
    """Trains a Random Forest model with GridSearchCV and saves evaluation metrics to CSV."""
    df = load_data()

    # Save original indices before vectorization
    df_indices = df.index

    # Feature extraction
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['cleaned_text'])
    y = df['sentiment']

    # Train-test split (preserve indices)
    X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
        X, y, df_indices, test_size=0.2, random_state=42
    )

    if grid_search:
        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
    
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

    # Create a DataFrame for the test set with predictions
    test_df = df.loc[test_idx].copy()  # Copy test set rows
    test_df['prediction'] = y_pred  # Add predictions


    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    # Connect to the database
    conn = sqlite3.connect(config.DATABASE_PATH)

    # saving predictions
    test_df.to_sql(config.PREDICTIONS_TABLE, conn, if_exists='replace', index=False)
    
    # saving grid search results
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_sql(config.EVALUATION_TABLE, conn,
                      if_exists='replace', index=False)
    # Commit and close the connection
    conn.commit()
    conn.close()
