import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle


app = Flask(__name__)
model = pickle.load(open('houseprediction.pkl','rb')) 

@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    
    '''
    For rendering results on HTML GUI
    '''
    home = float(request.args.get('home'))
    SqFt = float(request.args.get('SqFt'))
    bed = float(request.args.get('bed'))
    bath = float(request.args.get('bath'))
    offer = float(request.args.get('offers'))
    brick = float(request.args.get('brick'))
    neighborhood = float(request.args.get('neighborhood'))
    

    prediction = model.predict([[home,SqFt,bed,bath,offer,brick,neighborhood]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted house price : {}'.format(prediction))


if __name__ == "__main__":
    app.run()
