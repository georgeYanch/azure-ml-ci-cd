# model.py
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset and train a simple model
data = load_iris()
X = data.data
y = data.target
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save the trained model to a pickle file
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
