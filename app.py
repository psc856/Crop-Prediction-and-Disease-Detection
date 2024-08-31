from flask import Flask, request, render_template
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

# Load models and scalers
model = pickle.load(open('model.pkl', 'rb'))
sc = pickle.load(open('standscaler.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))
disease_model = tf.keras.models.load_model(r"D:\Temp\potatoes.h5")

# Define input shape expected by the disease model
INPUT_SHAPE = (256, 256, 3)
CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

# Create Flask app
app = Flask(__name__)

# Landing page route
@app.route('/')
def landing_page():
    return render_template("landing.html")

# Prediction page route
@app.route('/index')
def index():
    return render_template("crop_prediction.html")

@app.route('/predict', methods=['POST'])
def predict():
    N = request.form['Nitrogen']
    P = request.form['Phosporus']
    K = request.form['Potassium']
    temp = request.form['Temperature']
    humidity = request.form['Humidity']
    ph = request.form['Ph']
    rainfall = request.form['Rainfall']

    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    single_pred = np.array(feature_list).reshape(1, -1)

    scaled_features = ms.transform(single_pred)
    final_features = sc.transform(scaled_features)
    prediction = model.predict(final_features)

    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
        8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
        19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop to be cultivated with the provided data."
    return render_template('crop_prediction.html', result=result)

# Disease detection route
@app.route('/crop_disease', methods=['GET', 'POST'])
def crop_disease():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            image = Image.open(file)
            img_batch = preprocess_image(image)
            
            # Make prediction
            predictions = disease_model.predict(img_batch)
            predicted_class = CLASS_NAMES[np.argmax(predictions)]
            confidence = np.max(predictions) * 100

            return render_template("crop_disease.html", prediction=predicted_class, confidence=confidence)

    return render_template("crop_disease.html", prediction=None)

# Image preprocessing function
def preprocess_image(image):
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    image = image.resize((INPUT_SHAPE[1], INPUT_SHAPE[0]))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

# Main function to run the app
if __name__ == "__main__":
    app.run(debug=True)
