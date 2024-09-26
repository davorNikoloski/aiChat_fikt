from flask import Flask, render_template, request, jsonify
from llama_cpp import Llama
import time
import os
import json
from datetime import datetime

app = Flask(__name__)

# Load the large language model once when the app starts
LLM = Llama(model_path="./modals/mistral-7b-instruct-v0.1.Q5_K_M.gguf")

# To keep track of recent inputs and their timestamps
recent_inputs = []
DEBOUNCE_TIME = 2  # seconds
CHAT_HISTORY_DIR = './chat_history'

# Ensure chat history directory exists
os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('chatbot.html')

@app.route('/generate_response', methods=['POST'])
def generate_response():
    try:
        # Get the prompt from the frontend JSON data
        data = request.get_json()
        prompt = data.get('prompt', '').strip()

        # Check if the prompt is empty
        if not prompt:
            return jsonify({'response': "You haven't asked a question!"})

        # Check for repetitive inputs
        current_time = time.time()
        recent_inputs.append((prompt, current_time))
        recent_inputs[:] = [(p, t) for p, t in recent_inputs if current_time - t < DEBOUNCE_TIME]  # Clean up old entries

        if len(recent_inputs) > 1 and recent_inputs[-1][0] == recent_inputs[-2][0]:
            return jsonify({'response': "Please ask something else or rephrase your question."})

        print(f"Received prompt: {prompt}")

        # Set parameters for better response generation
        output = LLM(
            prompt=prompt,
            temperature=0.8,
            max_tokens=50,
            top_p=0.9,
            top_k=30  
        )
        
        print(output)

        # Get the response text
        response_text = output.get("choices", [{}])[0].get("text", "Sorry, I couldn't generate a response.")

        # Log the conversation to chat history
        log_conversation(prompt, response_text)

        return jsonify({'response': response_text})

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})

def log_conversation(prompt, response):
    # Generate a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(CHAT_HISTORY_DIR, f'chat_history_{timestamp}.json')

    # Create a conversation object
    conversation_data = {
        'timestamp': timestamp,
        'user_input': prompt,
        'bot_response': response
    }

    # Write the conversation data to a JSON file
    with open(filename, 'w') as f:
        json.dump(conversation_data, f, indent=4)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
