# hit endpoint to load model and generate response - MODEL IS LOADED, BUT STUCK IN GENERATING RESPONSE
import logging
import os
import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
# from google.colab import userdata
# from pyngrok import ngrok

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Note: This requires Hugging Face authorization

@app.route('/')
def home():
    return "Llama Model Inference Server is running!"

@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        # Check for Hugging Face token
        # hf_token = userdata.get('HUGGING_FACE_TOKEN')
        hf_token = os.environ.get('HUGGING_FACE_TOKEN')
        if not hf_token:
            raise ValueError("Hugging Face token is required. Set HUGGING_FACE_TOKEN environment variable.")

        print('Retrieving tokenizer')
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            use_fast=False  # Try disabling fast tokenizer
        )

        print('Retrieving model')
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
            token=hf_token,
            low_cpu_mem_usage=True  # Try to reduce memory usage
        )

        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Prepare the prompt
        prompt = data.get('prompt', '')
        if not prompt:
            return jsonify({'error': 'No prompt provided'}), 400
        print(prompt)

        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        print('Tokenizing input')

        # Generate text with more specific parameters
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=50,  # Use max_new_tokens instead of max_length
            num_return_sequences=1,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id  # Add pad token
        )
        print('Generating text')

        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(generated_text)
        print('Decoding generated text')

        return jsonify({
            'generated_text': generated_text
        })
    except Exception as e:
        print(f"Error loading model or generating text: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    # # Set ngrok authtoken
    # ngrok_authtoken = userdata.get('NGROK_AUTHTOKEN')
    # ngrok.set_auth_token(ngrok_authtoken)  # Replace with your ngrok authtoken

    # # Expose the port 8000
    # public_url = ngrok.connect(8000)
    # print(f"Public URL: {public_url}")

    # Run the Flask app
    app.run(host='0.0.0.0', port=8000, debug=True, use_reloader=False)