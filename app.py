import streamlit as st
import os
from utils import get_answer, text_to_speech, autoplay_audio, speech_to_text
from audio_recorder_streamlit import audio_recorder
from streamlit_float import *

# Float feature initialization
float_init()

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Ayush Sarkar. How may I assist you today?"}
        ]

initialize_session_state()

st.title("Ayush's Personal AI Assistant 🤖")
st.subheader("Powered by Together AI & DeepSeek-V3")

# Add some info about Ayush
with st.expander("💡 About Ayush"):
    st.markdown("""
    **Ayush Sarkar** - Recent BCA graduate from Sikkim Manipal Institute of Technology
    
    - 🎓 **Education**: BCA Graduate (June 2024)
    - 💪 **Interests**: AI/ML, Cooking, Travel, Photography, Fitness
    - 🚀 **Passion**: Determined problem solver who makes the impossible possible
    - 📱 **Projects**: Built interesting AI/ML projects during college
    
    Ask me anything about my background, skills, projects, or just chat!
    """)

# Create footer container for the microphone
footer_container = st.container()
with footer_container:
    st.markdown("🎤 **Click to record your message:**")
    audio_bytes = audio_recorder(
        text="",
        recording_color="#e74c3c",
        neutral_color="#34495e",
        icon_name="microphone",
        icon_size="2x"
    )

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Process audio input
if audio_bytes:
    with st.spinner("🎯 Processing your audio..."):
        webm_file_path = "temp_audio.mp3"
        with open(webm_file_path, "wb") as f:
            f.write(audio_bytes)
        
        # Use enhanced speech to text function
        transcript = speech_to_text(webm_file_path)
        
        if transcript:
            st.session_state.messages.append({"role": "user", "content": transcript})
            with st.chat_message("user"):
                st.write(transcript)
        else:
            st.warning("⚠️ Could not transcribe your audio. Please try speaking more clearly or check your microphone.")
        
        # Clean up temp file
        if os.path.exists(webm_file_path):
            os.remove(webm_file_path)

# Generate AI response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        # Get the AI response
        with st.spinner("Thinking..."):
            final_response = get_answer(st.session_state.messages)
        
        # Display the text response
        st.write(final_response)
        
        # FIXED: Generate and play audio response ONLY ONCE
        with st.spinner("🔊 Generating audio response..."):    
            audio_file = text_to_speech(final_response)
            if audio_file:
                autoplay_audio(audio_file)
                # Clean up audio file
                if os.path.exists(audio_file):
                    os.remove(audio_file)
        
        # Add response to messages
        st.session_state.messages.append({"role": "assistant", "content": final_response})

# Float the footer container
footer_container.float("bottom: 0rem;")

# Sidebar with conversation controls
with st.sidebar:
    st.header("💬 Conversation")
    
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = [
            {"role": "assistant", "content": "Hi! I'm Ayush Sarkar. How may I assist you today?"}
        ]
        st.rerun()
    
    st.markdown("---")
    
    st.header("📊 Chat Stats")
    total_messages = len(st.session_state.messages)
    user_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
    st.metric("Total Messages", total_messages)
    st.metric("Your Messages", user_messages)
    
    st.markdown("---")
    
    st.header("🎯 Sample Questions")
    sample_questions = [
        "What's your life story?",
        "What's your superpower?",
        "What areas do you want to grow in?",
        "Tell me about a challenge you overcame",
        "What misconceptions do people have about you?",
        "What projects have you worked on?",
        "What are your hobbies?"
    ]
    
    for question in sample_questions:
        if st.button(question, key=f"sample_{question[:20]}"):
            st.session_state.messages.append({"role": "user", "content": question})
            st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p><em>🚀 Ayush Sarkar's Personal AI Assistant</em></p>
    <p>Powered by Together AI • DeepSeek-V3 • Enhanced Speech Recognition</p>
</div>
""", unsafe_allow_html=True)
