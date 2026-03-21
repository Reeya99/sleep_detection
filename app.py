from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from flask import Response

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

    img = Image.open(file).convert('RGB')
    img = img.resize((224, 224))

    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction)
    result = classes[class_index]

    confidence = float(np.max(prediction)) * 100

    return render_template("index.html", 
                           prediction=result, 
                           confidence=round(confidence, 2))


import cv2

def gen_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Resize for model
        img = cv2.resize(frame, (224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)
        class_index = np.argmax(prediction)
        label = classes[class_index]
        confidence = np.max(prediction) * 100

        # Display on frame
        text = f"{label} ({confidence:.2f}%)"

        if "Closed" in label or "Yawning" in label:
            color = (0, 0, 255)  # RED
        else:
            color = (0, 255, 0)  # GREEN

        cv2.putText(frame, text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        

@app.route('/video')
def video():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
# Run app
if __name__ == "__main__":
    app.run(debug=True)