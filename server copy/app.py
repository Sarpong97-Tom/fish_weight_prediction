import flask
from flask import render_template,request,jsonify
import json
import numpy as np
import joblib


app = flask.Flask(__name__)

# def get_days_array (day):
#     if day.toLower() == "" 



def load_model(path):
    return joblib.load(path)


def make_predict(model,data):
    return model.predict(data)[0]

def get_desired_output():
    # init_list = [7,8,90.4,52.4,102.2,5.1,data['temp'],data['rh'],data['rain'],]
    return np.array([[1,2,3,33,4,4,5,6,7,8,9,0,4,1,4,5,6,1,1,1,0,1,1,1,0,1,1,0,0]])

def preprocess_data(data):
    temperature = data['temperature']
    rain = data['rain']
    day = data['day']
    month = data['month']
    humidity = data['humidity']
    print(temperature)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods = ['Post'])
def predict():
    from wsgi import model_path
    data = request.json
    results = make_predict(load_model(model_path),get_desired_output())
    return json.dumps({
        "results":results    
        })


if __name__ == "__main__":
    app.run('127.0.0.1',8000,debug = True)
