# save pretrained model - WORKS ON THE FIRST RUN
import os
import logging
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
# from pyngrok import ngrok
# from google.colab import userdata

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Flask app
app = Flask(__name__)

# Directories to save model and tokenizer
MODEL_DIR = "./model"
TOKENIZER_DIR = "./tokenizer"

# Model and tokenizer variables
model = None
tokenizer = None
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Requires Hugging Face authorization


def load_or_save_model():
    """
    Load the model and tokenizer during startup.
    If they do not exist locally, download and save them for subsequent runs.
    """
    global model, tokenizer

    # Check for Hugging Face token
    # hf_token = userdata.get("HUGGING_FACE_TOKEN")
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')
    if not hf_token:
        logger.error("Hugging Face token is required. Set HUGGING_FACE_TOKEN environment variable.")
        raise ValueError("Hugging Face token is required.")

    try:
        if not os.path.exists(MODEL_DIR) or not os.path.exists(TOKENIZER_DIR):
            logger.info("Downloading model and tokenizer from Hugging Face...")

            # Load and save tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                use_fast=False
            )
            tokenizer.save_pretrained(TOKENIZER_DIR)
            logger.info("Tokenizer saved locally.")

            # Load and save model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                token=hf_token,
                low_cpu_mem_usage=True
            )
            model.save_pretrained(MODEL_DIR)
            logger.info("Model saved locally.")
        else:
            # Load locally saved tokenizer and model
            logger.info("Loading tokenizer and model from local files...")
            tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_DIR,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info("Tokenizer and model loaded successfully from local files.")
    except Exception as e:
        logger.error(f"Failed to load or save model and tokenizer: {e}")
        raise e


@app.route("/")
def home():
    """
    Health check endpoint.
    """
    logger.info("Health check endpoint called.")
    return jsonify({"message": "Llama Model Inference Server is running!"})


@app.route("/generate", methods=["POST"])
def generate_text():
    """
    Generate text from the given prompt.
    """
    global model, tokenizer
    if model is None or tokenizer is None:
        logger.error("Model or tokenizer not loaded. Ensure startup initialization.")
        return jsonify({"error": "Model or tokenizer not loaded."}), 500

    data = request.json
    if not data or "prompt" not in data:
        logger.warning("No prompt provided in request.")
        return jsonify({"error": "Prompt is required."}), 400

    prompt = data["prompt"]
    logger.info(f"Received prompt: {prompt}")

    try:
        # Tokenize input
        logger.info("Tokenizing input...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Generate text
        logger.info("Generating text...")
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
        logger.info("Text generation complete.")

        # Decode output
        logger.info("Decoding generated text...")
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"Generated text: {generated_text}")

        return jsonify({"generated_text": generated_text})
    except Exception as e:
        logger.error(f"Error during text generation: {e}")
        return jsonify({"error": "An error occurred during text generation."}), 500

if __name__ == '__main__':
    # Load or save the model and tokenizer on startup
    logger.info("Initializing model and tokenizer...")
    try:
        load_or_save_model()
    except Exception as e:
        logger.error("Failed to initialize the model and tokenizer. Exiting.")
        raise e

    # # Set ngrok authtoken
    # ngrok_authtoken = userdata.get('NGROK_AUTHTOKEN')
    # ngrok.set_auth_token(ngrok_authtoken)  # Replace with your ngrok authtoken

    # # Expose the port 8000
    # public_url = ngrok.connect(8000)
    # print(f"Public URL: {public_url}")

    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)