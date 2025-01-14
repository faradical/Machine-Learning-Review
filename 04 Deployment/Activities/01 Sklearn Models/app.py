'''
This is a sample flask application demoing how model 
modularization improves the ability to use and deploy 
trained models.
'''

# Importing dependencies
from flask import Flask
from MyModule.Model import PipedModel, reshape_sample, reshape_feature

# Creating Flask application
app = Flask(__name__)

# Creating the home route
@app.route("/")
def home():
    return "<h1>Hello World!</h1>"

# Creating an api route usin the model
@app.route("/api/<X>")
def api(X):

    # Running a prediction on the route input
    pred = PipedModel.predict(reshape_sample(X))

    # Return the prediction to the user
    return f"{pred}"

# Running the app
if __name__ == '__main__':
    app.run(debug=True)