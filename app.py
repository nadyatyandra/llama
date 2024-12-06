import logging
import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Note: This requires Hugging Face authorization

try:
    # Check for Hugging Face token
    hf_token = os.environ.get('HUGGING_FACE_TOKEN')
    if not hf_token:
        raise ValueError("Hugging Face token is required. Set HUGGING_FACE_TOKEN environment variable.")
    logging.info('Finished retrieving hf_token')

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        token=hf_token,
    )
    logging.info('Finished retrieving tokenizer')
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map='auto',
        token=hf_token,
    )
    logging.info('Finished retrieving model')
except Exception as e:
    print(f"Error loading model: {e}")
    tokenizer = None
    model = None

@app.route('/')
def home():
    return "Llama Model Inference Server is running!"

@app.route('/generate', methods=['POST'])
def generate_text():
    if not model or not tokenizer:
        return jsonify({'error': 'Model not loaded'}), 500
    
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    prompt = data.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        logging.info('Finished tokenizing input')
        
        # Generate text
        outputs = model.generate(
            inputs.input_ids,
            max_length=50,  # Adjust for faster and smaller outputs
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True
        )
        logging.info('Finished generating text')
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logging.info('Finished decoding generated text')
        
        return jsonify({
            'generated_text': generated_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)