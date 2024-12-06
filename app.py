import torch
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"  # Note: This requires Hugging Face authorization

try:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=torch.float16,
    #     device_map='auto'
    # )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype=torch.float16,
        device_map='auto',
        use_auth_token=True
    )
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
        
        # Generate text
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7
        )
        
        # Decode generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return jsonify({
            'generated_text': generated_text
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)