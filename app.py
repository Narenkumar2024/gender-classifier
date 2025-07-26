# app.py
from flask import Flask, request, jsonify
from model import classifier, gender_features
import nltk

app = Flask(__name__)

@app.route('/')
def home():
    return "Gender Classifier is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    name = data.get("name", "")
    if not name:
        return jsonify({'error': 'Name not provided'}), 400
    gender = classifier.classify(gender_features(name))
    return jsonify({'name': name, 'gender': gender})

if __name__ == '__main__':
    nltk.download('names')  # ensure names corpus is downloaded
    app.run(debug=True)
