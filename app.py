import streamlit as st
import speech_recognition as sr
import numpy as np
import io
from together import Together
import os
import soundfile as sf
from gtts import gTTS
import tempfile
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json

# Initialize session state first
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "processing" not in st.session_state:
    st.session_state.processing = False

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

def transcribe_audio(audio_file):
    """Transcribe audio file to text using speech recognition"""
    if not audio_file:
        return ""
    
    r = sr.Recognizer()
    r.energy_threshold = 300
    r.dynamic_energy_threshold = True
    
    try:
        st.info("🎯 Processing and transcribing your audio...")
        
        # Create a temporary file to handle the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            # Write the uploaded file content to temp file
            audio_file.seek(0)  # Reset to beginning
            temp_file.write(audio_file.read())
            temp_file_path = temp_file.name
        
        # Convert audio to proper format using soundfile
        try:
            # Read the audio file
            data, samplerate = sf.read(temp_file_path)
            
            # Convert to WAV format that speech_recognition can handle
            wav_path = temp_file_path.replace('.wav', '_converted.wav')
            sf.write(wav_path, data, samplerate, format='WAV', subtype='PCM_16')
            
            # Use the converted WAV file for speech recognition
            with sr.AudioFile(wav_path) as source:
                r.adjust_for_ambient_noise(source, duration=0.5)
                audio = r.record(source)
            
            # Clean up temp files
            try:
                os.unlink(temp_file_path)
                os.unlink(wav_path)
            except:
                pass
                
        except Exception as conversion_error:
            st.warning(f"Audio conversion issue: {conversion_error}")
            # Fallback: try direct processing
            try:
                with sr.AudioFile(temp_file_path) as source:
                    r.adjust_for_ambient_noise(source, duration=0.5)
                    audio = r.record(source)
                os.unlink(temp_file_path)
            except Exception as fallback_error:
                st.error(f"Could not process audio file: {fallback_error}")
                try:
                    os.unlink(temp_file_path)
                except:
                    pass
                return ""

        # Try speech recognition
        try:
            text = r.recognize_google(audio, language='en-US')
            st.success("✅ Transcription successful!")
            return text.strip()
        except sr.UnknownValueError:
            st.warning("⚠️ Could not understand the audio clearly. Please try speaking louder and clearer.")
            return ""
        except sr.RequestError as e:
            st.error(f"❌ Speech recognition service error: {e}")
            return ""
            
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        return ""

def get_personalized_response(prompt):
    """Generate personalized response using Together API"""
    if not prompt.strip():
        return "I didn't catch your question clearly. Could you please ask again?"
    
    st.info("🤖 Generating Ayush's response...")
    
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
            temperature=0.7,
            timeout=30  # Add timeout
        )
        
        response_text = response.choices[0].message.content.strip()
        st.success("✅ Response generated!")
        return response_text
        
    except Exception as e:
        st.error(f"❌ Error communicating with Together API: {e}")
        return "Sorry, I'm having trouble processing that right now. Could you try asking your question again?"

def text_to_speech(text, lang='en'):
    """Convert text to speech and return audio file path"""
    clean_text = remove_emojis(text)
    
    if not clean_text.strip():
        st.warning("⚠️ No text content available for speech.")
        return None
    
    try:
        st.info("🔊 Generating audio response...")
        tts = gTTS(text=clean_text, lang=lang, slow=False)
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts.save(fp.name)
            audio_file_path = fp.name
            
        st.success("✅ Audio generated!")
        return audio_file_path
        
    except Exception as e:
        st.error(f"❌ Error generating speech: {e}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="Ayush's Interview Voice Bot", 
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🎤 Ayush's Interview Voice Bot")
st.subheader("AI Agent Position Interview - Home.LLC")

st.markdown("""
**Welcome to Ayush Sarkar's personalized interview bot!**

This bot will answer interview questions as Ayush would, drawing from his personal experiences, 
skills, and background. Perfect for practicing interview scenarios or getting to know Ayush better.

**How it works:**
1. Use the audio recorder below to ask your question
2. Wait for transcription and response generation
3. Listen to Ayush's personalized audio response
""")

with st.expander("💡 Sample Interview Questions"):
    st.markdown("""
    - What should we know about your life story?
    - What's your number one superpower?
    - What are the top 3 areas you'd like to grow in?
    - What misconceptions do your coworkers have about you?
    - How do you push your boundaries and limits?
    - Tell me about a challenging project you worked on?
    """)

# Main interview section
st.header("🎯 Interview Session")

# Audio input
st.markdown("### Record Your Question")
audio_file = st.audio_input("🎙️ Click to record your interview question")

# Process audio when uploaded
if audio_file is not None and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.container():
        st.markdown("---")
        
        # Step 1: Transcribe
        with st.spinner("Processing your audio..."):
            transcribed_text = transcribe_audio(audio_file)
        
        if transcribed_text:
            # Display question
            st.markdown("### 🤔 Your Question:")
            st.info(f'"{transcribed_text}"')
            
            # Step 2: Generate response
            with st.spinner("Ayush is thinking..."):
                ai_response = get_personalized_response(transcribed_text)
            
            # Display response
            st.markdown("### 💬 Ayush's Answer:")
            st.success(ai_response)
            
            # Step 3: Generate and play audio
            if ai_response:
                with st.spinner("Converting to speech..."):
                    audio_path = text_to_speech(ai_response)
                
                if audio_path:
                    st.markdown("### 🔊 Listen to Ayush's Response:")
                    try:
                        with open(audio_path, 'rb') as audio_file:
                            audio_bytes_response = audio_file.read()
                        st.audio(audio_bytes_response, format='audio/mp3')
                        
                        # Clean up temp file
                        if os.path.exists(audio_path):
                            try:
                                os.unlink(audio_path)
                            except:
                                pass
                    except Exception as e:
                        st.error(f"Error playing audio: {e}")
            
            # Save to conversation history
            st.session_state.conversation.append({
                "role": "interviewer", 
                "content": transcribed_text
            })
            st.session_state.conversation.append({
                "role": "ayush", 
                "content": ai_response
            })
            
            st.markdown("---")
            st.success("✅ Interview question completed! Record another question above.")
            
        else:
            st.warning("⚠️ Could not transcribe your audio. Please try again with clearer speech.")
    
    st.session_state.processing = False

# Sidebar - Interview History
if st.session_state.conversation:
    st.sidebar.header("📝 Interview History")
    
    # Group conversations by Q&A pairs
    qa_pairs = []
    for i in range(0, len(st.session_state.conversation), 2):
        if i + 1 < len(st.session_state.conversation):
            qa_pairs.append({
                'question': st.session_state.conversation[i]['content'],
                'answer': st.session_state.conversation[i + 1]['content']
            })
    
    # Display in reverse order (newest first)
    for idx, qa in enumerate(reversed(qa_pairs)):
        with st.sidebar.expander(f"Q{len(qa_pairs) - idx}: {qa['question'][:50]}..."):
            st.write(f"**Q:** {qa['question']}")
            st.write(f"**A:** {qa['answer']}")
    
    if st.sidebar.button("🗑️ Clear Interview History"):
        st.session_state.conversation = []
        st.rerun()

# Knowledge base viewer
with st.expander("📚 View Ayush's Knowledge Base"):
    st.json(PERSONAL_KNOWLEDGE_BASE)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><em>Ayush Sarkar - BCA Graduate - AI/ML Enthusiast - Ready for Home.LLC!</em></p>
    <p>🚀 Built with Streamlit • 🤖 Powered by Together AI • 🎙️ Speech Recognition</p>
</div>
""", unsafe_allow_html=True)
