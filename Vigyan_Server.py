from flask import Flask, request, jsonify
import google.generativeai as genai

# Configure the Google API key
genai.configure(api_key="AIzaSyAqoWwVloZ5N0RQ3PosmQgeUUb-Xe52YDw")

app = Flask(__name__)

# Set up the model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 100,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro-latest", generation_config=generation_config, safety_settings=safety_settings
)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Generate AI response
        convo = model.start_chat(history=[])
        response = convo.send_message(user_message)

        # Return the AI response
        return jsonify({"response": response.text})

    except Exception as e:
        # Log the exception and return an error response
        app.logger.error(f"Exception occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
