from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

app = Flask(__name__)

# Model details
MODEL_NAME = "MBZUAI/LaMini-T5-738M"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load tokenizer and model
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
print("Model and tokenizer loaded successfully.")

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    if not data or "input_text" not in data:
        return jsonify({"error": "Invalid input. Provide 'input_text'."}), 400

    input_text = data["input_text"]
    max_length = data.get("max_length", 50)  # Default length for the response

    try:
        # Tokenize input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate response using the model
        outputs = model.generate(
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
