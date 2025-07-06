import streamlit as st
from together import Together
import os
from gtts import gTTS
import tempfile
import base64
import speech_recognition as sr
import numpy as np
import soundfile as sf
from scipy import signal
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import concurrent.futures
import threading

# Initialize Together AI client
try:
    TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
except KeyError:
    st.error("TOGETHER_API_KEY not found in .streamlit/secrets.toml. "
             "Please create a file named 'secrets.toml' inside a folder named '.streamlit/' "
             "in your project directory, and add your API key like this:\n\n"
             "TOGETHER_API_KEY = \"your_actual_api_key_here\"")
    st.stop()

client = Together(api_key=TOGETHER_API_KEY)
DEEPSEEK_MODEL = "deepseek-ai/deepseek-V3"

# Personal Knowledge Base for RAG
PERSONAL_KNOWLEDGE_BASE = {
    "life_story": "I am Ayush Sarkar, I recently graduated in June from Sikkim Manipal Institute of Technology with a Bachelor's degree in Computer Applications. To speak about myself, I love to cook, I travel a lot and am very much interested in photography. I go to the gym regularly to balance a healthy lifestyle and look good. In this field of AI/ML, I have kept a keen interest for about 2 years now and I would say I learnt a lot on my own, from college, from friends and built some interesting projects applying those skills. I am proud to say that I am a very passionate learner for this field and when things interest me, I work hard to meet deadlines and learn as much as possible with every project.",
    
    "superpower": "My main superpower is to make something possible by hook or crook. If I don't know something, I would learn about it and use every tool possible. Maybe it will be outside my capabilities but I would always find ways to do it efficiently and bring out a solution every time I am faced with a challenge. That determination is my superpower.",
    
    "growth_areas": "Top 3 areas I would like to grow are: 1. AI/ML - since the start I have always wanted to learn more and more about this field and make my core so strong that I have knowledge from the depths and always have a solution in hand if I am faced with such problems. 2. Problem solving skills - In coding I want to be as efficient as I can be to improve the overall performance. I would like to give it my all to learn and solve problems as quickly as possible. 3. Project Management - I have some skills in project management as I have made a group project and I was the team leader, so planning and plotting what to do, when to do, how to do makes this thing really interesting as without a proper plan it seems vague and easy to get lost. I would love to get more hands-on real world experience and work as a team so that I get to learn more about management and cooperation.",
    
    "misconceptions": "People feel like I am chill and like to relax and be lazy, but when it comes to actual work I like to work hard and complete the task with utter determination. Another would be people might not think I am good at cooking, but they should try my white sauce chicken pasta.",
    
    "overcome_challenges": "When I was working on my college project as being the team leader, I had to make sure that everything was right from top to bottom with every presentation we had. I worked very late at night from afternoon till past midnight just to make sure that there's nothing wrong and it was all according to the requirements. I made sure that if my teammates made any mistake I would rectify them right away and inform them that there was a mistake and what it was so they are well informed and would know what not to do next. We had to make a report for every progress every month and present about it to all the faculties, so any mistake would cost us marks. With a lot of dedication, I made sure that the project at the end of the semester was a success."
}

# Pre-compute vectorizer and chunk vectors for faster RAG
@st.cache_resource
def initialize_rag_components():
    """Initialize and cache RAG components for faster retrieval"""
    chunks = []
    for key, value in PERSONAL_KNOWLEDGE_BASE.items():
        chunks.append(f"{key}: {value}")
    
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    chunk_vectors = vectorizer.fit_transform(chunks)
    
    return vectorizer, chunk_vectors, chunks

# Initialize RAG components once
vectorizer, chunk_vectors, knowledge_chunks = initialize_rag_components()

def find_relevant_context(query, top_k=3):
    """Find relevant context using pre-computed TF-IDF vectors - FASTER"""
    if not knowledge_chunks:
        return ""
    
    query_vector = vectorizer.transform([query])
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [knowledge_chunks[i] for i in top_indices if similarities[i] > 0.1]
    
    return "\n".join(relevant_chunks)

def get_answer(messages):
    """Get response from Together AI DeepSeek model with RAG - OPTIMIZED"""
    try:
        # Get the latest user message for RAG context
        latest_message = messages[-1]["content"] if messages else ""
        relevant_context = find_relevant_context(latest_message)
        
        # Create system prompt with RAG context
        system_prompt = f"""You are Ayush Sarkar, speaking naturally in first person about yourself.
        
        Here is your personal information to draw from:
        {relevant_context}
        
        Instructions:
        - Respond as Ayush naturally and conversationally
        - Be authentic, professional, and show your personality
        - Use the provided personal context to give genuine answers
        - Add appropriate humor where it fits naturally
        - Keep responses concise but informative (2-4 sentences typically)
        - Don't use emojis in your response
        - Avoid repeatedly mentioning your degree or college unless specifically asked about education
        - Focus on the substance of what you're sharing rather than credentials
        - If asked about something not in your background, be honest but relate it to your learning mindset
        - Sound like a real person having a conversation, not giving a resume
        """
        
        # Format messages for the API
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        # OPTIMIZATION: Reduced max_tokens and timeout for faster response
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=formatted_messages,
            max_tokens=300,  # Reduced from 400
            temperature=0.7,
            timeout=20  # Reduced from 30
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Error getting response from Together AI: {e}")
        return "Sorry, I'm having trouble processing that right now. Could you try asking your question again?"

# Cache compiled regex for faster emoji removal
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
    """Remove emojis from text - OPTIMIZED with cached regex"""
    emoji_pattern = get_emoji_pattern()
    return emoji_pattern.sub(r'', text).strip()

def text_to_speech(text):
    """Convert text to speech using gTTS - OPTIMIZED"""
    try:
        # Remove emojis from text
        clean_text = remove_emojis(text)
        
        if not clean_text.strip():
            return None
        
        # OPTIMIZATION: Use faster TTS settings
        tts = gTTS(text=clean_text, lang='en', slow=False, tld='com')  # Added tld for faster processing
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
            
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def autoplay_audio(file_path):
    """Auto-play audio in Streamlit - OPTIMIZED"""
    if not file_path or not os.path.exists(file_path):
        return
        
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            # OPTIMIZATION: Simplified HTML audio tag
            md = f'<audio controls autoplay="true"><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>'
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def speech_to_text(audio_file_path):
    """Enhanced speech to text with multiple fallback strategies - OPTIMIZED"""
    if not audio_file_path or not os.path.exists(audio_file_path):
        return ""

    temp_files_to_clean_up = []

    try:
        # Read audio file
        with open(audio_file_path, 'rb') as f:
            audio_bytes = f.read()

        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
            temp_files_to_clean_up.append(temp_audio_path)

        try:
            # Enhanced audio preprocessing
            data, samplerate = sf.read(temp_audio_path)

            # Ensure it's mono
            if len(data.shape) > 1:
                data = np.mean(data, axis=1)

            # Audio enhancement steps
            # 1. Remove DC offset
            data = data - np.mean(data)

            # 2. Normalize with headroom
            if np.max(np.abs(data)) > 0:
                data = data / np.max(np.abs(data)) * 0.9

            # 3. Resample to 16kHz if needed
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

            # 4. Apply pre-emphasis filter
            pre_emphasis = 0.97
            data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])

            processed_path = temp_audio_path.replace('.wav', '_processed.wav')
            sf.write(processed_path, data, samplerate)
            temp_files_to_clean_up.append(processed_path)

        except Exception as e:
            processed_path = temp_audio_path

        # Initialize recognizer - OPTIMIZED settings
        r = sr.Recognizer()
        r.energy_threshold = 200
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.5  # Reduced from 0.6
        r.operation_timeout = 10  # Reduced from 15

        # OPTIMIZATION: Try fewer languages for faster processing
        languages_to_try = ['en-US', 'en-IN']  # Reduced from 5 to 2 most relevant

        with sr.AudioFile(processed_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.05)  # Reduced from 0.1
            audio_data = r.record(source)

        for lang in languages_to_try:
            try:
                text = r.recognize_google(audio_data, language=lang)
                if text and text.strip():
                    return text.strip()
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue

        # OPTIMIZATION: Only one fallback strategy instead of multiple
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
                r_amp.adjust_for_ambient_noise(source, duration=0.05)
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
        # Clean up temp files
        for path in temp_files_to_clean_up:
            try:
                if os.path.exists(path):
                    os.unlink(path)
            except:
                pass
