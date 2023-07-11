import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# Set the random seed for reproducibility
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

def load_data(file_path):
    """
    Load data from a CSV file and preprocess it.
    """
    df = pd.read_csv(file_path)
    df = df.drop_duplicates()
    df = pd.get_dummies(df, columns=['country', 'gender'])
    
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Split data into training and testing sets.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """
    Train a logistic regression model and return it.
    """
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
    return lr_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on a testing set and return the accuracy score.
    """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def save_model(model, file_path):
    """
    Save a trained model to a file.
    """
    joblib.dump(model, file_path)
    

def main():
    # Load and preprocess the data
    X, y = load_data(r'C:\Users\mussie\Music\final pro\Bank Customer Churn Prediction.csv')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train a logistic regression model
    lr_model = train_model(X_train, y_train)
    
    # Evaluate the model on the testing set
    accuracy = evaluate_model(lr_model, X_test, y_test)
    print("Accuracy:", accuracy)
    
    # Save the model to a file
    save_model(lr_model, "model_2.joblib")

if __name__ == "__main__":
    main()