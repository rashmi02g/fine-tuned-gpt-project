import os
import requests
import torch
from flask import Flask, request, jsonify
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Function to download the model file from Google Drive
def download_file_from_google_drive(url, destination):
    response = requests.get(url)
    with open(destination, 'wb') as f:
        f.write(response.content)

# URL to the model file on Google Drive
model_url = 'https://drive.google.com/uc?export=download&id=1YO8HaM79UFYDTLTYbk9-Y1eZ-Fyvb5Dk'
model_path = 'fine-tuned-gpt2/model.safetensors'

# Download the model if it does not exist
if not os.path.exists(model_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    download_file_from_google_drive(model_url, model_path)

# Load the GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.load_state_dict(torch.load(model_path, map_location='cpu'))  # Use map_location='cpu' for CPU inference
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define the endpoint for generating text
@app.route('/generate', methods=['POST'])
def generate_text():
    try:
        data = request.get_json()
        prompt = data.get('prompt', '')

        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400

        # Encode the input text and generate a response
        input_ids = tokenizer.encode(prompt, return_tensors='pt')
        output = model.generate(input_ids, max_length=100, num_return_sequences=1)

        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

        return jsonify({'response': generated_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
