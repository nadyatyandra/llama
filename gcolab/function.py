# use function to load model and generate response - WORKING
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from google.colab import userdata

logging.basicConfig(level=logging.INFO)

# Load model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"

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

    # Prepare the prompt
    prompt = 'Explain quantum computing'

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

except Exception as e:
    print(f"Error loading model or generating text: {e}")
    import traceback
    traceback.print_exc()