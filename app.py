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
    """Enhanced transcribe audio file to text using speech recognition with multiple fallbacks"""
    if not audio_file:
        return ""
    
    try:
        st.info("🎯 Processing your audio...")
        
        # Handle UploadedFile object from st.audio_input
        if hasattr(audio_file, 'read'):
            audio_bytes = audio_file.read()
            audio_file.seek(0)
        else:
            audio_bytes = audio_file
        
        # Debug info
        st.info(f"Audio file size: {len(audio_bytes)} bytes")
        
        # Save to temporary file first (more reliable approach)
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
            temp_audio.write(audio_bytes)
            temp_audio_path = temp_audio.name
        
        try:
            # Enhanced audio preprocessing
            try:
                data, samplerate = sf.read(temp_audio_path)
                st.info(f"Original audio: {samplerate}Hz, {len(data)} samples, duration: {len(data)/samplerate:.1f}s")
                
                # Ensure it's mono
                if len(data.shape) > 1:
                    data = np.mean(data, axis=1)
                
                # Audio enhancement steps
                # 1. Remove DC offset
                data = data - np.mean(data)
                
                # 2. Normalize with headroom
                if np.max(np.abs(data)) > 0:
                    data = data / np.max(np.abs(data)) * 0.7  # Leave some headroom
                
                # 3. Simple noise gate (remove very quiet sections)
                noise_threshold = np.max(np.abs(data)) * 0.02
                data = np.where(np.abs(data) < noise_threshold, 0, data)
                
                # 4. Resample to 16kHz if needed (optimal for speech recognition)
                target_sr = 16000
                if samplerate != target_sr:
                    # Use simple resampling if scipy not available
                    try:
                        from scipy import signal
                        data = signal.resample(data, int(len(data) * target_sr / samplerate))
                        samplerate = target_sr
                        st.info(f"Resampled to {target_sr}Hz for better recognition")
                    except ImportError:
                        # Fallback: simple decimation/interpolation
                        if samplerate > target_sr:
                            # Downsample
                            factor = int(samplerate // target_sr)
                            data = data[::factor]
                            samplerate = samplerate // factor
                        st.info(f"Basic resampling to ~{samplerate}Hz")
                
                # 5. Apply simple pre-emphasis filter (boost high frequencies)
                pre_emphasis = 0.97
                data = np.append(data[0], data[1:] - pre_emphasis * data[:-1])
                
                # Save enhanced audio
                processed_path = temp_audio_path.replace('.wav', '_processed.wav')
                sf.write(processed_path, data, samplerate)
                st.info("✅ Audio preprocessing complete")
                
            except Exception as e:
                st.warning(f"Audio processing error: {e}, using original file")
                processed_path = temp_audio_path
            
            # Initialize recognizer with enhanced settings
            r = sr.Recognizer()
            
            # Optimized recognizer settings
            r.energy_threshold = 200  # Lower threshold for quiet audio
            r.dynamic_energy_threshold = True
            r.dynamic_energy_adjustment_damping = 0.15
            r.dynamic_energy_ratio = 1.5
            r.pause_threshold = 0.6  # Shorter pause threshold
            r.operation_timeout = 10  # Prevent infinite hanging
            r.phrase_threshold = 0.3
            r.non_speaking_duration = 0.5
            
            # Load and adjust audio
            with sr.AudioFile(processed_path) as source:
                st.info("🎧 Loading and analyzing audio...")
                
                # More aggressive noise adjustment
                r.adjust_for_ambient_noise(source, duration=min(1.0, len(data)/samplerate/3))
                
                # Record with specific settings
                audio_data = r.record(source)
                st.info("✅ Audio loaded successfully!")
            
            # Multiple transcription attempts with different strategies
            transcription_results = []
            
            # Strategy 1: Standard Google with multiple languages
            st.info("🔍 Attempting Google Speech Recognition...")
            languages_to_try = ['en-US', 'en-IN', 'en-GB', 'en-AU', 'en']
            
            for lang in languages_to_try:
                try:
                    text = r.recognize_google(audio_data, language=lang)
                    if text and text.strip():
                        confidence_score = 0.8  # Assume high confidence for successful recognition
                        transcription_results.append((text.strip(), confidence_score, f"Google-{lang}"))
                        st.success(f"✅ Google ({lang}): '{text}'")
                        break  # Stop on first success
                except sr.UnknownValueError:
                    continue
                except sr.RequestError as e:
                    st.warning(f"Google {lang} service error: {e}")
                    continue
            
            # Strategy 2: Google with show_all for confidence scores
            if not transcription_results:
                st.info("🔍 Trying detailed Google recognition...")
                try:
                    result = r.recognize_google(audio_data, language='en-US', show_all=True)
                    if result and 'alternative' in result:
                        for alt in result['alternative']:
                            if 'transcript' in alt and alt['transcript'].strip():
                                confidence = alt.get('confidence', 0.5)
                                transcription_results.append((alt['transcript'].strip(), confidence, "Google-detailed"))
                                st.info(f"Google alternative (confidence {confidence:.2f}): '{alt['transcript']}'")
                except Exception as e:
                    st.info(f"Detailed Google recognition failed: {e}")
            
            # Strategy 3: Try with different audio processing
            if not transcription_results:
                st.info("🔍 Trying with audio amplification...")
                try:
                    # Amplify audio and try again
                    amplified_data = data * 2.0
                    amplified_data = np.clip(amplified_data, -1.0, 1.0)
                    
                    amplified_path = temp_audio_path.replace('.wav', '_amplified.wav')
                    sf.write(amplified_path, amplified_data, samplerate)
                    
                    with sr.AudioFile(amplified_path) as source:
                        r.adjust_for_ambient_noise(source, duration=0.2)
                        audio_data_amp = r.record(source)
                    
                    text = r.recognize_google(audio_data_amp, language='en-US')
                    if text and text.strip():
                        transcription_results.append((text.strip(), 0.7, "Google-amplified"))
                        st.success(f"✅ Amplified audio recognition: '{text}'")
                    
                    # Clean up
                    if os.path.exists(amplified_path):
                        os.unlink(amplified_path)
                        
                except Exception as e:
                    st.info(f"Amplified audio attempt failed: {e}")
            
            # Strategy 4: Try Whisper-like preprocessing (if available)
            if not transcription_results:
                st.info("🔍 Trying advanced audio preprocessing...")
                try:
                    # Apply more aggressive noise reduction
                    # Simple spectral subtraction approach
                    from scipy.signal import butter, filtfilt
                    
                    # High-pass filter to remove low-frequency noise
                    nyquist = samplerate / 2
                    low_cutoff = 80 / nyquist
                    high_cutoff = (samplerate / 2 - 1) / nyquist
                    
                    b, a = butter(4, [low_cutoff, high_cutoff], btype='band')
                    filtered_data = filtfilt(b, a, data)
                    
                    # Normalize again
                    filtered_data = filtered_data / np.max(np.abs(filtered_data)) * 0.8
                    
                    filtered_path = temp_audio_path.replace('.wav', '_filtered.wav')
                    sf.write(filtered_path, filtered_data, samplerate)
                    
                    with sr.AudioFile(filtered_path) as source:
                        r.adjust_for_ambient_noise(source, duration=0.1)
                        audio_data_filt = r.record(source)
                    
                    text = r.recognize_google(audio_data_filt, language='en-US')
                    if text and text.strip():
                        transcription_results.append((text.strip(), 0.6, "Google-filtered"))
                        st.success(f"✅ Filtered audio recognition: '{text}'")
                    
                    # Clean up
                    if os.path.exists(filtered_path):
                        os.unlink(filtered_path)
                        
                except ImportError:
                    st.info("Advanced filtering not available (scipy not installed)")
                except Exception as e:
                    st.info(f"Filtered audio attempt failed: {e}")
            
            # Select best result
            if transcription_results:
                # Sort by confidence score
                best_result = max(transcription_results, key=lambda x: x[1])
                st.success(f"🎯 Best transcription ({best_result[2]}, confidence: {best_result[1]:.2f}): '{best_result[0]}'")
                return best_result[0]
            
            # If all methods failed, provide detailed debugging info
            st.error("❌ All transcription methods failed.")
            st.markdown("### 🔧 Debug Information:")
            st.write(f"• **Audio duration:** {len(data)/samplerate:.1f} seconds")
            st.write(f"• **Sample rate:** {samplerate}Hz")
            st.write(f"• **Audio level:** {np.max(np.abs(data)):.3f}")
            st.write(f"• **RMS level:** {np.sqrt(np.mean(data**2)):.3f}")
            st.write(f"• **Zero crossing rate:** {np.mean(np.abs(np.diff(np.sign(data)))):.3f}")
            
            # Audio quality assessment
            if len(data)/samplerate < 1.0:
                st.warning("⚠️ Audio is very short (< 1 second). Try speaking for longer.")
            elif np.max(np.abs(data)) < 0.1:
                st.warning("⚠️ Audio level is very low. Try speaking louder or closer to the microphone.")
            elif np.sqrt(np.mean(data**2)) < 0.01:
                st.warning("⚠️ Audio appears to be mostly silence. Check your microphone.")
            
            # Provide helpful suggestions
            st.markdown("### 💡 Troubleshooting Tips:")
            st.markdown("""
            - **Speak clearly and loudly** into your microphone
            - **Reduce background noise** as much as possible
            - **Use a good quality microphone** or headset
            - **Speak for at least 2-3 seconds**
            - **Try the manual text input** as a fallback
            """)
            
            return "FALLBACK_TO_TEXT_INPUT"
            
        finally:
            # Clean up all temp files
            temp_files = [
                temp_audio_path,
                temp_audio_path.replace('.wav', '_processed.wav'),
                temp_audio_path.replace('.wav', '_amplified.wav'),
                temp_audio_path.replace('.wav', '_filtered.wav')
            ]
            
            for path in temp_files:
                try:
                    if os.path.exists(path):
                        os.unlink(path)
                except:
                    pass
                    
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        st.error("Please try recording again or use the text input option.")
        return "FALLBACK_TO_TEXT_INPUT"
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
            # Check for fallback to text input
            if transcribed_text == "FALLBACK_TO_TEXT_INPUT":
                st.markdown("### ⌨️ Type Your Question Instead:")
                manual_text = st.text_input("Enter your interview question:", key="manual_question")
                if manual_text:
                    transcribed_text = manual_text
                    st.info(f'Using typed question: "{transcribed_text}"')
                else:
                    st.session_state.processing = False
                    st.stop()
            
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
