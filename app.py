import streamlit as st
import speech_recognition as sr
import soundfile as sf
import sounddevice as sd
import numpy as np
import io
from together import Together
import os
from gtts import gTTS
import tempfile
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

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

PERSONAL_KNOWLEDGE_BASE = {
    "life_story": "I am Ayush Sarkar ,i recently graduated in June from Sikkim Manipal Institute of Technology as a BCA student. To speak about myself I love to cook, I travel a lot and very much interested in photography , I go to the gym regularly to balance a healthy lifestyle and look good. In this field of AI/ML i have kept a keen interest since 2 years or so and I would say i learnt a lot from my own from the college from friends and built some interesting projects applying those skills and I am proud to say that i am a very passionate learner for this field and things which interest me i work hard to meet deadlines and learn with every project as much as possible.",
    
    "superpower": "My main super power is to make something possible by hook or crook if i don't know something i would learn about it use every tool possible maybe it will be outside my capabilities but i would always make out ways to do it efficiently and bring out a solution every time i am faced with a challenge , that determination is my super power.",
    
    "growth_areas": "Top 3 areas i would like to grow are - 1. AI/ML since the start i have always wanted to learn more and more about this filed and make my core so strong that i have knowledge from depths and always have a solution in hand if I am faced with such problems. 2. problem solving skills - In code i want to be as efficient as i can be to improve the overall performance i would like give it my all to learn and solve problems as quickly as possible. 3. Project Management - I have some skills in project management as i have made a group project and i was the team leader so planning and plotting what to do when to do how do makes this thing really interesting as without a proper plan it seems vague and easy to get lost. I would love get more hands on real world experience and work as a team so that i get to learn more about management and cooperation.",
    
    "misconceptions": "People feel like i am chill and like to relax be lazy but when it comes to actual work i like to work hard and complete the task with utter determination. Another would be people might not think i am good at cooking but they should try my white sauce chicken pasta.",
    
    "overcome_challenges": "When i was working on my college project as being the team leader i had to make sure that everything was right from top to bottom with every presentation we had, i worked very late at night from afternoon till past midnight just to make sure that there's nothing wrong and it was all according to the requirements, i made sure that if my teammates made any mistake i would rectify them right away and inform that there was a mistake and what was it so they are well informed and would know what not to do next. We had to make a report for every progress every month and present about it to all the faculties so any mistake would cost us marks so with a lot of dedication i made sure that the project at the end of the semester was a success."
}

def create_knowledge_chunks():
    chunks = []
    
    for key, value in PERSONAL_KNOWLEDGE_BASE.items():
        chunks.append(f"{key}: {value}")
    
    return chunks

def find_relevant_context(query, top_k=3):
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

def remove_emojis(text):
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

def record_audio(duration=5, fs=44100):
    st.info(f"Recording for {duration} seconds...")
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with st.spinner("Listening..."):
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='int16')
        
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / duration)
            status_text.text(f"Recording... {i + 1}/{duration} seconds")
        
        sd.wait()
    
    progress_bar.empty()
    status_text.empty()
    st.success("Recording complete!")
    return recording, fs

def transcribe_audio(audio_data, sample_rate):
    r = sr.Recognizer()
    
    audio_data_float = audio_data.astype(np.float32) / np.iinfo(np.int16).max
    buffer = io.BytesIO()
    sf.write(buffer, audio_data_float, sample_rate, format='WAV')
    buffer.seek(0)

    with sr.AudioFile(buffer) as source:
        audio = r.record(source)

    try:
        st.info("Transcribing audio...")
        text = r.recognize_google(audio)
        st.success("Transcription complete!")
        return text
    except sr.UnknownValueError:
        st.error("Speech Recognition could not understand audio. Please try again.")
        return ""
    except sr.RequestError as e:
        st.error(f"Could not request results from Google Web Speech API service; check your internet connection: {e}")
        return ""

def get_personalized_response(prompt):
    st.info("Getting personalized response...")
    
    relevant_context = find_relevant_context(prompt)
    
    system_prompt = f"""You are Ayush Sarkar, a recent BCA graduate from Sikkim Manipal Institute of Technology. 
    You are being interviewed for an AI Agent position at Home.LLC.
    
    Here is your personal information to draw from:
    {relevant_context}
    
    Instructions:
    - Answer as Ayush Sarkar in first person
    - Be conversational, professional, and show personality
    - Use the provided personal context to give authentic answers
    - Add appropriate humor where it fits naturally
    - Keep responses concise but informative (2-3 sentences max)
    - Don't use emojis in your response
    - If asked about something not in your background, be honest but relate it to your learning mindset
    """
    
    try:
        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            max_tokens=250,
            temperature=0.7
        )
        st.success("Response generated!")
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error communicating with Together API: {e}")
        return "Sorry, I couldn't process that right now. Could you try asking again?"

def text_to_speech(text, lang='en'):
    clean_text = remove_emojis(text)
    
    if not clean_text.strip():
        st.warning("No text content available for speech after cleaning.")
        return None
    
    with st.spinner("Generating audio response..."):
        try:
            tts = gTTS(text=clean_text, lang=lang, slow=False)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                tts.save(fp.name)
                audio_file_path = fp.name
            st.success("Audio generated!")
            return audio_file_path
        except Exception as e:
            st.error(f"Error generating speech: {e}")
            return None

def process_voice_interaction(duration):
    audio_data, fs = record_audio(duration=duration)
    
    if audio_data is not None:
        transcribed_text = transcribe_audio(audio_data, fs)
        
        if transcribed_text:
            st.write(f"**Question:** {transcribed_text}")
            
            ai_response = get_personalized_response(transcribed_text)
            st.write(f"**Ayush's Answer:** {ai_response}")
            
            st.session_state.conversation.append({"role": "interviewer", "content": transcribed_text})
            st.session_state.conversation.append({"role": "ayush", "content": ai_response})
            
            if ai_response:
                audio_path = text_to_speech(ai_response)
                if audio_path:
                    st.audio(audio_path, format='audio/mp3', start_time=0, loop=False, autoplay=True)
                    
                    if os.path.exists(audio_path):
                        try:
                            os.unlink(audio_path)
                        except:
                            pass
            
            return True
        else:
            st.warning("No clear speech was detected. Please try speaking louder or clearer.")
            return False
    
    return False

st.set_page_config(page_title="Ayush's Interview Voice Bot", layout="centered")

st.title("Ayush's Interview Voice Bot")
st.subheader("AI Agent Position Interview - Home.LLC")

st.markdown("""
**Welcome to Ayush Sarkar's personalized interview bot!**

This bot will answer interview questions as Ayush would, drawing from his personal experiences, 
skills, and background. Perfect for practicing interview scenarios or getting to know Ayush better.

**How it works:**
1. Click "Start Interview Question" 
2. Ask any interview question (life story, superpowers, growth areas, etc.)
3. Listen to Ayush's personalized response
""")

st.info("""
**Sample Interview Questions:**
- What should we know about your life story?
- What's your number one superpower?
- What are the top 3 areas you'd like to grow in?
- What misconceptions do your coworkers have about you?
- How do you push your boundaries and limits?
""")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

st.header("Interview Session")

col1, col2 = st.columns([3, 1])

with col1:
    recording_duration = st.slider("Recording duration (seconds):", 3, 15, 7)

if st.button("Start Interview Question", key="interview_button", type="primary"):
    st.markdown("---")
    
    with st.container():
        success = process_voice_interaction(recording_duration)
        
        if success:
            st.markdown("---")
            st.success("Interview question completed successfully!")
        else:
            st.error("Please try again with a clearer question.")

if st.session_state.conversation:
    st.sidebar.header("Interview History")
    
    for i, entry in enumerate(reversed(st.session_state.conversation)):
        if entry["role"] == "interviewer":
            st.sidebar.markdown(f"**Q{len(st.session_state.conversation)//2 - i//2}:** {entry['content']}")
        else:
            st.sidebar.markdown(f"**Ayush:** {entry['content']}")
        
        if i < len(st.session_state.conversation) - 1 and entry["role"] == "interviewer":
            st.sidebar.markdown("---")
    
    if st.sidebar.button("Clear Interview History"):
        st.session_state.conversation = []
        st.experimental_rerun()

with st.expander("View Ayush's Knowledge Base"):
    st.json(PERSONAL_KNOWLEDGE_BASE)

st.markdown("---")
st.markdown("*Ayush Sarkar - BCA Graduate - AI/ML Enthusiast - Ready for Home.LLC!*")