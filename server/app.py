import flask
from flask import render_template,request,jsonify
import json
import numpy as np
import joblib


app = flask.Flask(__name__)


fish = {
    "Bream":[1,0,0,0,0,0,0],
    "Roach":[0,1,0,0,0,0,0],
    "Whitefish":[0,0,1,0,0,0,0],
    "Parkki":[0,0,0,1,0,0,0],
    "Perch":[0,0,0,0,1,0,0],
    "Pike":[0,0,0,0,0,1,0],
    "Pike":[0,0,0,0,0,0,1],
}


def get_species_array(species):
    return fish[species]



def load_model(path):
    return joblib.load(path)


def make_predict(model,data):
    return model.predict(data)[0]


def preprocess_data(data):
    length1 = float(data['length1'])
    length2 = float(data['length2'])
    length3 = float(data['length3'])
    height = float(data['height'])
    width = float(data['width'])
    species = data['species']
    init_array = [length1,length2,length3,height,width] +get_species_array(species)
    print(init_array)
    return np.array([init_array])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['Post'])
def predict():
    from wsgi import model_path
    # model_path = "../training/model/regressor.pkl"
    data = request.json
    print(preprocess_data(data))
    results = make_predict(load_model(model_path),preprocess_data(data))
    return json.dumps({
        "results":results  
        })


if __name__ == "__main__":
    app.run('127.0.0.1',8000,debug = True)
