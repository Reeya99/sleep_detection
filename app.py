from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

app = Flask(__name__)

# Load model
model = load_model("drowsiness_model.h5")

# Class labels
classes = ['Closed_Eyes', 'Open_Eyes', 'Yawning', 'No_Yawning']

# Home route
@app.route('/')
def home():
    return render_template("index.html")

# Predict route
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    # Open and preprocess image
    img = Image.open(file).convert('RGB')   # ensure 3 channels
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    return render_template("index.html", prediction=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)