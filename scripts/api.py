from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("models/model.pkl")

@app.route('/predict', methods = ['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'churn_prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(port=5000, debug=True)

    