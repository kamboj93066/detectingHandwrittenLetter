
import os
from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow import keras

app = Flask(__name__)


# Define the image size and path to your trained model
img_height, img_width = 28, 28
model_path = '/Users/ishantkamboj/Documents/Chandigarh_university/SEM_5/AML/Exp/Experiment_7/model'  # Update with the correct path

# Load your trained model
model = keras.models.load_model(model_path)

# Define a function to make predictions
def predict_alphabet(image_path):
    # Load and preprocess the image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Ensure RGB format
    img = cv2.resize(img, (img_height, img_width))
    img = img / 255.0  # Normalize pixel values to [0, 1]

    # Reshape the image to match the model's input shape
    img = np.reshape(img, (1, img_height, img_width, 3))

    # Make a prediction
    prediction = model.predict(img)

    # Get the predicted class (alphabet letter)
    predicted_class_index = np.argmax(prediction)

    # Map the class index back to the alphabet letter
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"  # Assuming 26 classes (A to Z)
    predicted_alphabet = alphabet[predicted_class_index]

    return predicted_alphabet




@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the uploaded image temporarily
            file_path = "uploaded_image.png"
            file.save(file_path)
            predicted_alphabet = predict_alphabet(file_path)
            return f"Predicted Alphabet: {predicted_alphabet}"
    return render_template('index.html')

if __name__ == '__main':
    app.run(debug=True)
