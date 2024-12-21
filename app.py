# app.py
from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the data from the POST request
        data = request.get_json()
        prediction_input = np.array(data["input"]).reshape(1, -1)
        
        # Get prediction from the model
        prediction = model.predict(prediction_input)
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)