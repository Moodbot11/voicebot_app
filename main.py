import openai
import os
import pyaudio
import wave
from pydub import AudioSegment
from pydub.playback import play
import streamlit as st

# Ensure your OpenAI API key is set in your environment
openai.api_key = os.getenv('OPENAI_API_KEY')

# Configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 5
assistant_id = "asst_73h87uzHn59aUalZCquIjama"

# Initialize PyAudio
p = pyaudio.PyAudio()

def record_audio(filename, duration=RECORD_SECONDS):
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    frames = []

    st.write('Recording...')
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    stream.stop_stream()
    stream.close()

    st.write('Finished recording')

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def transcribe_audio(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        transcription = openai.Audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    return transcription['text']

def generate_response(transcribed_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
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
