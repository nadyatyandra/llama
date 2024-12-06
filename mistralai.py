from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Flask app setup
app = Flask(__name__)

# Model and tokenizer (global variables)
model_name = "mistralai/Mistral-7B-Instruct-v0.2"  # Using Mistral as an open-source alternative
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map='auto'
    )
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    tokenizer = None
    model = None

@app.route('/')
def home():
    return "Mistral AI Model Inference Server is running!"

# Generate endpoint
@app.route("/generate", methods=["POST"])
def generate_text():
    # Validate model is loaded
    if not model or not tokenizer:
        return jsonify({"error": "Model not loaded"}), 500

    # Validate request JSON
    data = request.get_json()
    if not data or "prompt" not in data:
        return jsonify({"error": "No prompt provided"}), 400

    prompt = data["prompt"]

    try:
        # Prepare the prompt in Mistral's instruction format
        full_prompt = f"<s>[INST] {prompt} [/INST]"

        # Tokenize input
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)

        # Generate text
        outputs = model.generate(
            inputs.input_ids,
            max_length=250,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            top_p=0.95
        )

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main block
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
