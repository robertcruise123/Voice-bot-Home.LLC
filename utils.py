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
    "life_story": "I am Ayush Sarkar ,i recently graduated in June from Sikkim Manipal Institute of Technology as a BCA student. To speak about myself I love to cook, I travel a lot and very much interested in photography , I go to the gym regularly to balance a healthy lifestyle and look good. In this field of AI/ML i have kept a keen interest since 2 years or so and I would say i learnt a lot from my own from the college from friends and built some interesting projects applying those skills and I am proud to say that i am a very passionate learner for this field and things which interest me i work hard to meet deadlines and learn with every project as much as possible.",
    
    "superpower": "My main super power is to make something possible by hook or crook if i don't know something i would learn about it use every tool possible maybe it will be outside my capabilities but i would always make out ways to do it efficiently and bring out a solution every time i am faced with a challenge , that determination is my super power.",
    
    "growth_areas": "Top 3 areas i would like to grow are - 1. AI/ML since the start i have always wanted to learn more and more about this filed and make my core so strong that i have knowledge from depths and always have a solution in hand if I am faced with such problems. 2. problem solving skills - In code i want to be as efficient as i can be to improve the overall performance i would like give it my all to learn and solve problems as quickly as possible. 3. Project Management - I have some skills in project management as i have made a group project and i was the team leader so planning and plotting what to do when to do how do makes this thing really interesting as without a proper plan it seems vague and easy to get lost. I would love get more hands on real world experience and work as a team so that i get to learn more about management and cooperation.",
    
    "misconceptions": "People feel like i am chill and like to relax be lazy but when it comes to actual work i like to work hard and complete the task with utter determination. Another would be people might not think i am good at cooking but they should try my white sauce chicken pasta.",
    
    "overcome_challenges": "When i was working on my college project as being the team leader i had to make sure that everything was right from top to bottom with every presentation we had, i worked very late at night from afternoon till past midnight just to make sure that there's nothing wrong and it was all according to the requirements, i made sure that if my teammates made any mistake i would rectify them right away and inform that there was a mistake and what was it so they are well informed and would know what not to do next. We had to make a report for every progress every month and present about it to all the faculties so any mistake would cost us marks so with a lot of dedication i made sure that the project at the end of the semester was a success."
}

def create_knowledge_chunks():
    """Create chunks from personal knowledge base"""
    chunks = []
    for key, value in PERSONAL_KNOWLEDGE_BASE.items():
        chunks.append(f"{key}: {value}")
    return chunks

def find_relevant_context(query, top_k=3):
    """Find relevant context using TF-IDF similarity"""
    chunks = create_knowledge_chunks()
    
    if not chunks:
        return ""
    
    vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
    chunk_vectors = vectorizer.fit_transform(chunks)
    query_vector = vectorizer.transform([query])
    
    similarities = cosine_similarity(query_vector, chunk_vectors)[0]
    
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    relevant_chunks = [chunks[i] for i in top_indices if similarities[i] > 0.1]
    
    return "\n".join(relevant_chunks)

def get_answer(messages):
    """Get response from Together AI DeepSeek model with RAG"""
    try:
        # Get the latest user message for RAG context
        latest_message = messages[-1]["content"] if messages else ""
        relevant_context = find_relevant_context(latest_message)
        
        # Create system prompt with RAG context
        system_prompt = f"""You are Ayush Sarkar, a recent BCA graduate from Sikkim Manipal Institute of Technology.
        
        Here is your personal information to draw from:
        {relevant_context}
        
        Instructions:
        - Answer as Ayush Sarkar in first person
        - Be conversational, professional, and show personality
        - Use the provided personal context to give authentic answers
        - Add appropriate humor where it fits naturally
        - Keep responses concise but informative
        - Don't use emojis in your response
        - If asked about something not in your background, be honest but relate it to your learning mindset
        """
        
        # Format messages for the API
        formatted_messages = [{"role": "system", "content": system_prompt}]
        formatted_messages.extend(messages)
        
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=formatted_messages,
            max_tokens=500,
            temperature=0.7,
            timeout=30
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        st.error(f"Error getting response from Together AI: {e}")
        return "Sorry, I'm having trouble processing that right now. Could you try asking your question again?"

def text_to_speech(text):
    """Convert text to speech using gTTS"""
    try:
        # Remove emojis from text
        clean_text = remove_emojis(text)
        
        if not clean_text.strip():
            return None
            
        tts = gTTS(text=clean_text, lang='en', slow=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            return fp.name
            
    except Exception as e:
        st.error(f"Error generating speech: {e}")
        return None

def autoplay_audio(file_path):
    """Auto-play audio in Streamlit"""
    if not file_path or not os.path.exists(file_path):
        return
        
    try:
        with open(file_path, "rb") as f:
            data = f.read()
            b64 = base64.b64encode(data).decode()
            md = f"""
                <audio controls autoplay="true">
                <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
                </audio>
                """
            st.markdown(md, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error playing audio: {e}")

def remove_emojis(text):
    """Remove emojis from text"""
    emoji_pattern = re.compile("["
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
    
    return emoji_pattern.sub(r'', text).strip()

def speech_to_text(audio_file_path):
    """Enhanced speech to text with multiple fallback strategies"""
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

        # Initialize recognizer
        r = sr.Recognizer()
        r.energy_threshold = 200
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.6
        r.operation_timeout = 15

        # Multiple transcription strategies
        transcription_results = []

        # Strategy 1: Standard Google with multiple languages
        languages_to_try = ['en-US', 'en-IN', 'en-GB', 'en-AU', 'en']

        with sr.AudioFile(processed_path) as source:
            r.adjust_for_ambient_noise(source, duration=0.1)
            audio_data = r.record(source)

        for lang in languages_to_try:
            try:
                text = r.recognize_google(audio_data, language=lang)
                if text and text.strip():
                    transcription_results.append((text.strip(), 0.8, f"Google-{lang}"))
                    break
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue

        # Strategy 2: Try with amplification if first strategy failed
        if not transcription_results:
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
                    r_amp.adjust_for_ambient_noise(source, duration=0.1)
                    audio_data_amp = r_amp.record(source)

                text = r_amp.recognize_google(audio_data_amp, language='en-US')
                if text and text.strip():
                    transcription_results.append((text.strip(), 0.7, "Google-amplified"))

            except:
                pass

        # Return best result
        if transcription_results:
            best_result = max(transcription_results, key=lambda x: x[1])
            return best_result[0]

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