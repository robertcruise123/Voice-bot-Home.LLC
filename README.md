Step by Step guide to use the voice bot :
📋 What You'll Build
A conversational AI voice bot that can:
- Answer questions about Ayush Sarkar (using voice)
- Speak responses back to you
- Work through your web browser

🛠️ Step 1: Install Required Software
Install Python
1. Go to [python.org](https://python.org/downloads/) and download Python 3.8+
2. IMPORTANT: Check "Add Python to PATH" during installation
3. Verify: Open terminal and type `python --version`

Install Git
1. Go to [git-scm.com](https://git-scm.com/downloads) and install Git
2. Verify: Open terminal and type `git --version`

📁 Step 2: Clone the Repository
bash
# Clone the project
git clone [YOUR_REPOSITORY_URL_HERE]
# Navigate to the folder
cd voice-bot

🔑 Step 3: Setup API Key

1. Sign up at [together.ai](https://together.ai) and get your API key
2. Rename `secrets.toml.example` to `secrets.toml` in the `.streamlit` folder
3. Replace the placeholder with your actual API key:
toml
TOGETHER_API_KEY = "your_actual_api_key_here"

🚀 Step 4: Install & Run

bash
# Install dependencies
pip install -r requirements.txt
# Run the voice bot
streamlit run main.py
Your browser will open automatically at `http://localhost:8501`

🎯 How to Use
Voice Chat  
- Click the microphone button at the bottom
- Speak your question clearly
- The bot will respond with both text and voice
 Try These Questions:
- "What should we know about your life story?"
- "What's your superpower?"
- "Tell me about your growth areas"
  
🎉 You're Done!
