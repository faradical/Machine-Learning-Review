# Import Dependencies
import numpy as np
import pickle
from sklearn.pipeline import Pipeline


# Creating reshape functions
def reshape_feature(X):
    return X.reshape(-1, 1)

def reshape_sample(X):
    return np.array(X).reshape(1, -1)


# Import pipeline steps
file = open('Scaler.pkl', 'rb')
scaler = pickle.load(file)

file = open('PolynomialFeatures.pkl', 'rb')
polyfeatures = pickle.load(file)

file = open('Linear Regression.pkl', 'rb')
loaded_model = pickle.load(file)

file.close()


# Defining the pipeline steps
steps = [
    ("Scaler", scaler),
    ("PolynomialFeatures", polyfeatures),
    ("Linear Regression", loaded_model)
]


# Defining a new pipeline with the loaded model
PipedModel = Pipeline(steps)


# Testing code.
# Only executes if file is run directly, not as a module
if __name__ == "__main__":
    print(PipedModel.predict(reshape_sample(200)))