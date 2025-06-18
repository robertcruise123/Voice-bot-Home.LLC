import streamlit as st
import os
import speech_recognition as sr
import numpy as np
import soundfile as sf
import tempfile
import re
from scipy import signal
from utils import get_answer, text_to_speech, autoplay_audio
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Float feature initialization
float_init()

# Cache compiled regex for better performance
@st.cache_data
def get_emoji_pattern():
    """Cache compiled emoji regex pattern"""
    return re.compile("["
                     u"\U0001F600-\U0001F64F"  # emoticons
                     u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                     u"\U0001F680-\U0001F6FF"  # transport & map symbols
                     u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                     u"\U00002500-\U00002BEF"  # chinese char
                     u"\U00002702-\U000027B0"
                     u"\U00002702-\U000027B0"
                     u"\U000024C2-\U0001F251"
                     u"\U0001f926-\U0001f937"
                     u"\U00010000-\U0010ffff"
                     u"\u2640-\u2642" 
                     u"\u2600-\u2B55"
                     u"\u200d"
                     u"\u23cf"
                     u"\u23e9"
                     u"\u231a"
                     u"\ufe0f"  # dingbats
                     u"\u3030"
                     "]+", flags=re.UNICODE)

def remove_emojis(text):
    """Remove emojis from text - OPTIMIZED"""
    emoji_pattern = get_emoji_pattern()
    return emoji_pattern.sub(r'', text).strip()

def enhanced_speech_to_text(audio_file_path):
    """OPTIMIZED transcribe audio file to text - Faster processing"""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return ""

    temp_files_to_clean_up = []

    try:
        # Read audio file
        audio_bytes = open(audio_file_path, 'rb').read()

        # Save to temporary file first
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
            temp_files_to_clean_up.append(temp_audio_path)

        try:
            # OPTIMIZED audio preprocessing - fewer steps for speed
            data, samplerate = sf.read(temp_audio_path)

            # Ensure it's mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Essential audio enhancement only
            data = data - np.mean(data)  # Remove DC offset

            # Normalize
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data)) * 0.9

            # Resample to 16kHz if needed
            target_sr = 16000
            if samplerate != target_sr:
                try:
                    data = signal.resample(data, int(len(data) * target_sr / samplerate))
                    samplerate = target_sr
                except ImportError:
                    if samplerate > target_sr:
                        factor = int(samplerate // target_sr)
                        data = data[::factor]
                        samplerate = samplerate // factor

            # Simple pre-emphasis
            pre_emphasis = 0.97
            data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

            processed_path = temp_audio_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, data, samplerate)
            temp_files_to_clean_up.append(processed_path)

        except Exception as e:
            processed_path = temp_audio_path

        # OPTIMIZED recognizer settings for speed
        r = sr.Recognizer()
        r.energy_threshold = 200
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.5  # Reduced
        r.operation_timeout = 10  # Reduced from 15
        r.phrase_threshold = 0.3
        r.non_speaking_duration = 0.4  # Reduced

        # Load and adjust audio
        with sr.AudioFile(processed_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.05)  # Faster noise adjustment
            audio_data = r.record(source)

        # OPTIMIZATION: Try fewer languages for faster processing
        languages_to_try = ['en-US', 'en-IN']  # Reduced from 5 to 2

        for lang in languages_to_try:
            try:
                text = r.recognize_google(audio_data, language=lang)
                if text and text.strip():
                    return text.strip()
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                continue

        # OPTIMIZATION: Only one fallback strategy
        try:
            result = r.recognize_google(audio_data, language='en-US', show_all=True)
            if result and 'alternative' in result:
                for alt in result['alternative']:
                    if 'transcript' in alt and alt['transcript'].strip():
                        return alt['transcript'].strip()
        except:
            pass

        # Final fallback with amplification
        try:
            data_amp, samplerate_amp = sf.read(temp_audio_path)
            if len(data_amp.shape) > 1:
                data_amp = np.mean(data_amp, axis=1)
            
            data_amp = data_amp - np.mean(data_amp)
            if np.max(np.abs(data_amp)) > 0:
                data_amp = data_amp / np.max(np.abs(data_amp)) * 0.9

            amplified_path = temp_audio_path.replace('.wav', '_amplified.wav')
            sf.write(amplified_path, data_amp, samplerate_amp)
            temp_files_to_clean_up.append(amplified_path)

            with sr.AudioFile(amplified_path) as source:
                r_amp = sr.Recognizer()
                r_amp.energy_threshold = 100
                r_amp.adjust_for_ambient_noise(source, duration=0.05)  # Reduced
                audio_data_amp = r_amp.record(source)

            text = r_amp.recognize_google(audio_data_amp, language='en-US')
            if text and text.strip():
                return text.strip()

        except:
            pass

        return ""

    except Exception as e:
        return ""

    finally:
        # Clean up all temp files
        for path in temp_files_to_clean_up:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! How may I assist you today?"}
        ]

initialize_session_state()

st.title("OpenAI Conversational Chatbot 🤖")

# Sample Interview Questions Dropdown
st.subheader("💡 Sample Interview Questions")
sample_questions = [
    "Select a question...",
    "What should we know about your life story in a few sentences?",
    "What's your #1 superpower?",
    "What are the top 3 areas you'd like to grow in?",
    "What misconception do your coworkers have about you?",
    "How do you push your boundaries and limits?"
]

selected_question = st.selectbox(
    "Choose a sample question to ask:",
    sample_questions,
    key="sample_question_selector"
)

# Add the selected question to chat when user selects one
if selected_question != "Select a question..." and selected_question:
    if st.button("Ask this question", key="ask_sample_question"):
        st.session_state.messages.append({"role": "user", "content": selected_question})
        st.rerun()

st.divider()

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    audio_bytes = audio_recorder()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if audio_bytes:
    # Write the audio bytes to a file
    with st.spinner("Transcribing..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        
        # Use enhanced speech to text function
        transcript = enhanced_speech_to_text(webm_file_path)
        
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
        else:
            st.warning("⚠️ Could not transcribe your audio. Please try speaking more clearly or check your microphone.")
        
        # Clean up temp file
        if os.path.exists(webm_file_path):
            os.remove(webm_file_path)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # OPTIMIZATION: Faster spinner messages
        with st.spinner("Thinking..."):
            final_response = get_answer(st.session_state.messages)
        with st.spinner("Generating audio..."):    
            audio_file = text_to_speech(final_response)
            autoplay_audio(audio_file)
        st.write(final_response)
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        if audio_file and os.path.exists(audio_file):
            os.remove(audio_file)

# Float the footer container and provide CSS to target it with
footer_container.float("bottom: 0rem;")
