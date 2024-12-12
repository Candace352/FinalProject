from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file
import numpy as np
import pandas as pd
import joblib
import requests
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import matplotlib.pyplot as plt
import io
import os

from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__)

# Database Configuration
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///db.sqlite"
app.config["SECRET_KEY"] = "abcd123"
db = SQLAlchemy()

login_manager = LoginManager()
login_manager.init_app(app)

# User model
class Users(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(250), unique=True, nullable=False)
    password = db.Column(db.String(250), nullable=False)

db.init_app(app)

# Create tables
with app.app_context():
    db.create_all()

@login_manager.user_loader
def loader_user(user_id):
    return Users.query.get(user_id)

# Route for the login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = Users.query.filter_by(username=request.form.get("username")).first()
        if user and user.password == request.form.get("password"):
            login_user(user)
            return render_template('weather.html'))
        else:
            return "Invalid credentials!"
    return render_template('login.html')

# Route for the register page
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        user = Users(username=request.form.get("username"),
                     password=request.form.get("password"))
        db.session.add(user)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template('register.html')

# Load models and scaler
model = load_model('./wmodel.keras', custom_objects={'mse': MeanSquaredError()})
scaler = joblib.load('./scaler.pkl')
predicted_values = []

# Weather API Key
WEATHER_API_KEY = '823435d1ab0542fd9f4185847240312'

# Directory to store generated files
UPLOAD_FOLDER = 'downloads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Chatbot Model Configuration
MODEL_NAME = "MBZUAI/LaMini-T5-738M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model for chatbot
print("Loading chatbot model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
chatbot_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
print("Chatbot model and tokenizer loaded successfully.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/landing')
def landing():
    return render_template('landing.html')

@app.route('/fetch_weather', methods=['POST'])
def fetch_weather():
    location = request.json.get('location')
    if not location:
        return jsonify({'error': 'Location is required'}), 400

    url = f"http://api.weatherapi.com/v1/current.json?key={WEATHER_API_KEY}&q={location}"
    response = requests.get(url)

    if response.status_code == 200:
        weather_data = response.json()
        temperature = weather_data['current']['temp_c']
        humidity = weather_data['current']['humidity']
        rainfall = weather_data['current']['precip_mm']
        return jsonify({
            'temperature': temperature,
            'humidity': humidity,
            'rainfall': rainfall,
        })
    else:
        return jsonify({'error': 'Failed to fetch weather data. Please check the location or API key.'}), 500

@app.route('/predict_water_availability', methods=['POST'])
def predict_water_availability():
    global predicted_values
    data = request.json['data']
    input_array = np.array(data).reshape((1, 1, len(data)))
    predictions = model.predict(input_array).flatten()
    predicted_values = predictions.tolist()
    return jsonify({'predictions': predicted_values})

@app.route('/plot_predictions', methods=['POST'])
def plot_predictions():
    global predicted_values
    if not predicted_values:
        return jsonify({'error': 'No predictions available. Please make predictions first.'}), 400

    # Plotting the predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(predicted_values) + 1), predicted_values, marker='o', label="Predicted Values")
    ax.set_title("Predicted Water Availability")
    ax.set_xlabel("Days")
    ax.set_ylabel("Groundwater Level (m)")
    ax.grid(True)
    ax.legend()

    # Save the plot to a BytesIO buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    plt.close(fig)
    img.seek(0)

    return send_file(img, mimetype='image/png')

@app.route('/download_summary', methods=['POST'])
def download_summary():
    global predicted_values
    if not predicted_values:
        return jsonify({'error': 'No predictions available to download.'}), 400

    # Save predicted summary to a text file
    summary_file_path = os.path.join(UPLOAD_FOLDER, 'prediction_summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write("Predicted Water Availability:\n")
        f.write('\n'.join([f"Day {i+1}: {value}" for i, value in enumerate(predicted_values)]))

    return send_file(summary_file_path, as_attachment=True)

@app.route('/download_graph', methods=['POST'])
def download_graph():
    global predicted_values
    if not predicted_values:
        return jsonify({'error': 'No predictions available to plot and download.'}), 400

    # Plotting the predictions
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(predicted_values) + 1), predicted_values, marker='o', label="Predicted Values")
    ax.set_title("Predicted Water Availability")
    ax.set_xlabel("Days")
    ax.set_ylabel("Groundwater Level (m)")
    ax.grid(True)
    ax.legend()

    # Save the plot to a PNG file
    graph_file_path = os.path.join(UPLOAD_FOLDER, 'predicted_graph.png')
    plt.savefig(graph_file_path, format='png')
    plt.close(fig)

    return send_file(graph_file_path, as_attachment=True)

@app.route('/predict', methods=['GET', 'POST'])
def predictions():
    if request.method == 'POST':
        data = request.json
        if not data or "input_text" not in data:
            return jsonify({"error": "Invalid input. Provide 'input_text'."}), 400

        input_text = data["input_text"]

        try:
            # Perform prediction logic
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = chatbot_model.generate(
                inputs["input_ids"],
                max_length=50,
                num_beams=5,
                early_stopping=True
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({"response": response})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('predictions.html')


@app.route('/weather')
def weather():
    return render_template('weather.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        data = request.json
        if not data or "input_text" not in data:
            return jsonify({"error": "Invalid input. Provide 'input_text'."}), 400

        input_text = data["input_text"]
        max_length = data.get("max_length", 50)  # Default length for the response

        try:
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

            # Generate response using the model
            outputs = chatbot_model.generate(
                inputs["input_ids"],
                max_length=max_length,
                num_beams=5,
                early_stopping=True
            )

            # Decode the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return jsonify({"response": response})

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    return render_template('chat.html')

if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
