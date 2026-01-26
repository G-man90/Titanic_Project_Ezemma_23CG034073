import os
from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler (saved together as a tuple)
model, scaler = joblib.load("model/titanic_survival_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            pclass = int(request.form["pclass"])
            sex = int(request.form["sex"])
            age = float(request.form["age"])
            fare = float(request.form["fare"])
            embarked = int(request.form["embarked"])

            data = np.array([[pclass, sex, age, fare, embarked]])
            data = scaler.transform(data)

            result = model.predict(data)[0]
            prediction = "Survived" if int(result) == 1 else "Did Not Survive"

        except Exception:
            prediction = "Invalid input. Please check your values."

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
