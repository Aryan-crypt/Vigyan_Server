from flask import Flask, request, jsonify
import google.generativeai as genai
from gtts import gTTS
import os
from io import BytesIO
import pygame
import threading

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

stop_flag = threading.Event()  # Create a flag to signal when to stop playback

def play_audio(audio_fp):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(audio_fp)
        pygame.mixer.music.play()

        # Wait until the music finishes playing or the stop flag is set
        while pygame.mixer.music.get_busy() and not stop_flag.is_set():
            pass

        pygame.mixer.music.stop()
    except Exception as e:
        app.logger.error(f"Error during audio playback: {e}")

@app.route('/chat', methods=['POST'])
def chat():
    global stop_flag
    stop_flag.clear()  # Reset the stop flag before starting a new playback

    try:
        data = request.json
        user_message = data.get('message')

        if not user_message:
            return jsonify({"error": "No message provided"}), 400

        # Generate AI response
        convo = model.start_chat(history=[])
        response = convo.send_message(user_message)

        # Convert the response to speech
        text_response = response.text
        tts = gTTS(text_response, lang='en')
        
        # Save audio file to a BytesIO object
        audio_fp = BytesIO()
        tts.write_to_fp(audio_fp)
        audio_fp.seek(0)

        # Play the audio in a separate thread so it can be stopped
        audio_thread = threading.Thread(target=play_audio, args=(audio_fp,))
        audio_thread.start()

        # Return the AI response
        return jsonify({"response": text_response})

    except Exception as e:
        # Log the exception and return an error response
        app.logger.error(f"Exception occurred: {e}")
        return jsonify({"error": "Internal server error"}), 500

@app.route('/stop', methods=['POST'])
def stop_audio():
    global stop_flag
    stop_flag.set()  # Set the stop flag to signal that playback should stop
    return jsonify({"message": "Playback stopped"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
