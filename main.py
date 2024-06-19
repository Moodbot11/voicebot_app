import sounddevice as sd
import soundfile as sf
import openai
import os
from pydub import AudioSegment
from pydub.playback import play
import streamlit as st

# Ensure your OpenAI API key is set in your environment
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration
RATE = 44100
RECORD_SECONDS = 5

# Use your assistant ID here
assistant_id = "asst_73h87uzHn59aUalZCquIjama"

def record_audio(filename, duration=RECORD_SECONDS):
    st.write('Recording...')
    audio = sd.rec(int(duration * RATE), samplerate=RATE, channels=1, dtype='int16')
    sd.wait()
    sf.write(filename, audio, RATE)
    st.write('Finished recording')

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription['text']

def generate_response(transcribed_text):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": transcribed_text}
        ],
        max_tokens=150
    )
    return response.choices[0].message['content'].strip()

def text_to_speech(generated_text, output_file_path):
    response = openai.Audio.create(
        model="tts-1",
        input=generated_text,
        voice="alloy"
    )
    with open(output_file_path, "wb") as f:
        for chunk in response.iter_bytes():
            f.write(chunk)
    st.write(f"Speech file saved to {output_file_path}")

def play_audio(filename):
    audio_segment = AudioSegment.from_file(filename, format="mp3")
    play(audio_segment)

def main():
    st.title("Voice Bot App")

    if st.button("Record"):
        audio_file_path = "input_audio.wav"
        record_audio(audio_file_path, duration=5)

        transcribed_text = transcribe_audio(audio_file_path)
        st.write(f"Transcribed text: {transcribed_text}")

        generated_text = generate_response(transcribed_text)
        st.write(f"Generated response: {generated_text}")

        response_speech_file_path = "response_output.mp3"
        text_to_speech(generated_text, response_speech_file_path)

        play_audio(response_speech_file_path)

if __name__ == "__main__":
    main()
