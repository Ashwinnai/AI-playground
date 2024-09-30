import streamlit as st
from typing import Generator
from groq import Groq
import fitz  # PyMuPDF for PDF
import pandas as pd
import docx
from moviepy.editor import VideoFileClip
import pytesseract
from PIL import Image
import plotly.express as px
import time  # Importing time module to measure response time
import streamlit_shadcn_ui as ui  # Streamlit Shadcn UI for buttons and other elements

# Set Streamlit page configuration
st.set_page_config(page_icon="💬", layout="wide", page_title="Ashwin's LLM")

def icon_with_text(emoji: str, text: str):
    """Shows an emoji and text as a Notion-style page header."""
    st.write(
        f'<div style="display: flex; align-items: center;">'
        f'<span style="font-size: 78px; line-height: 1; margin-right: 15px;">{emoji}</span>'
        f'<h1 style="font-size: 48px; line-height: 1.2;">{text}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Use the icon_with_text function to display the icon and the AI Playground title
icon_with_text("🏎️", "Ashwin's AI Playground")

# Sidebar for options with Shadcn UI elements
st.sidebar.title("Options")
st.sidebar.header("Ashwin's AI Playground")
st.sidebar.write("Welcome to the AI Playground! Here, you can experiment with different AI models and functionalities.")

# Sidebar dropdown for API key input
with st.sidebar.expander("Enter Password"):
    st.subheader("Enter Password")
    api_key = st.text_input("Enter your Password ", type="password")

# Store API key in session state immediately if provided
if api_key:
    st.session_state.api_key = api_key

# Initialize session state for api_key if not present
if "api_key" not in st.session_state:
    st.session_state.api_key = None

# Sidebar for file upload with enhanced UI
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF, Word, Excel, or Video file", 
    type=["pdf", "docx", "xlsx", "mp4", "avi", "mov"]
)

# Sidebar for topics
st.sidebar.subheader("Select a Topic")
topic = st.sidebar.selectbox("Choose a topic", ["General", "Data Analysis", "Machine Learning", "Natural Language Processing"])

# Define extraction functions
def extract_text_from_pdf(file):
    doc = fitz.open(file)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_word(file):
    doc = docx.Document(file)
    text = ""
    for para in doc.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    text = df.to_string()
    return text

def extract_text_from_video(file):
    video = VideoFileClip(file)
    frames = video.iter_frames()
    text = ""
    for frame in frames:
        img = Image.fromarray(frame)
        text += pytesseract.image_to_string(img)
    return text

def extract_text_from_uploaded_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_word(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return extract_text_from_excel(file)
    elif file.type.startswith("video"):
        return extract_text_from_video(file)
    else:
        return "Unsupported file type."

# Initialize chat history and selected model in session state if not already present
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Sidebar for model selection
models = {
    "distil-whisper-large-v3-en": {"name": "Distil-Whisper English", "tokens": 25000, "developer": "HuggingFace", "type": "File (Audio)", "max_file_size": "25 MB"},
    "gemma2-9b-it": {"name": "Gemma 2 9B", "tokens": 8192, "developer": "Google", "type": "Text"},
    "gemma-7b-it": {"name": "Gemma 7B", "tokens": 8192, "developer": "Google", "type": "Text"},
    "llama3-groq-70b-8192-tool-use-preview": {"name": "LLaMA3 Groq 70B Tool Use (Preview)", "tokens": 8192, "developer": "Groq", "type": "Text"},
    "llama3-groq-8b-8192-tool-use-preview": {"name": "LLaMA3 Groq 8B Tool Use (Preview)", "tokens": 8192, "developer": "Groq", "type": "Text"},
    "llama-3.1-70b-versatile": {"name": "LLaMA 3.1 70B", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llama-3.2-1b-preview": {"name": "LLaMA 3.2 1B (Preview)", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llama-3.2-3b-preview": {"name": "LLaMA 3.2 3B (Preview)", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llama-3.2-11b-vision-preview": {"name": "LLaMA 3.2 11B Vision (Preview)", "tokens": 8192, "developer": "Meta", "type": "Text and Vision"},
    "llama-3.2-90b-vision-preview": {"name": "LLaMA 3.2 90B (Preview)", "tokens": 8192, "developer": "Meta", "type": "Text and Vision"},
    "llama-guard-3-8b": {"name": "LLaMA Guard 3 8B", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llava-v1.5-7b-4096-preview": {"name": "LLaVA 1.5 7B", "tokens": 4096, "developer": "Haotian Liu", "type": "Text"},
    "llama3-70b-8192": {"name": "Meta LLaMA 3 70B", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "llama3-8b-8192": {"name": "Meta LLaMA 3 8B", "tokens": 8192, "developer": "Meta", "type": "Text"},
    "mixtral-8x7b-32768": {"name": "Mixtral 8x7B", "tokens": 32768, "developer": "Mistral", "type": "Text"},
    "whisper-large-v3": {"name": "Whisper", "tokens": 25000, "developer": "OpenAI", "type": "File (Audio)", "max_file_size": "25 MB"}
}

# Default to llama-3.1-8b-instant
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=list(models.keys()).index("llama-3.1-8b-instant")  # Default to llama-3.1-8b-instant
)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

max_tokens_range = models[model_option]["tokens"]

st.sidebar.subheader("Max Tokens")
# Adjust max_tokens slider dynamically based on the selected model
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,  # Minimum value to allow some flexibility
    max_value=max_tokens_range,
    # Default value set to 7680
    value=min(7680, max_tokens_range),
    step=512,
    help=f"Adjust the maximum number of tokens (words) for the model's response. Max for selected model: {max_tokens_range}"
)

# Extract text from uploaded file and display it
knowledge_base = ""
if uploaded_file:
    knowledge_base = extract_text_from_uploaded_file(uploaded_file)
    st.write("Extracted text from uploaded file:")
    st.write(knowledge_base)

# Sidebar buttons for Scroll to Top and Clear Chat
if st.sidebar.button("Scroll to top"):
    st.rerun()

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# Initialize Groq client if API key is provided
if "client" not in st.session_state and st.session_state.api_key:
    st.session_state.client = Groq(api_key=st.session_state.api_key)

# Display chat messages from history on app rerun
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])

def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# Function to handle the chat input and generate a response
def handle_chat_input(prompt):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(prompt)

    # Fetch response from Groq API
    full_response = ""

    # Start time
    start_time = time.time()  # Capture start time before generating the response

    try:
        chat_completion = st.session_state.client.chat.completions.create(
            model=model_option,
            messages=[
                {
                    "role": m["role"],
                    "content": m["content"]
                }
                for m in st.session_state.messages
            ] + [{"role": "system", "content": knowledge_base}],
            max_tokens=max_tokens,
            stream=True
        )

        # Use the generator function with st.write_stream
        with st.chat_message("assistant", avatar="🤖"):
            chat_responses_generator = generate_chat_responses(chat_completion)
            full_response = "".join(chat_responses_generator)

    except Exception as e:
        st.error(e, icon="🚨")

    # Calculate the response time
    response_time = time.time() - start_time  # Calculate response time

    # Append the full response to session_state.messages
    if full_response:
        # Add response time to the end of the assistant's message
        full_response += f"\n\n_Response generated in {response_time:.2f} seconds._"

        st.session_state.messages.append(
            {"role": "assistant", "content": full_response})

    # Display the response time using a Streamlit metric in the sidebar
    st.sidebar.metric(label="Response Time (seconds)", value=f"{response_time:.2f}")

    # Trigger a rerun to update the UI dynamically
    st.rerun()

# Chat input section
prompt = st.chat_input("Enter your prompt here...")
if prompt:
    handle_chat_input(prompt)

# Footer with enhanced design
st.markdown("""
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center; padding: 10px; background-color: #00000;">
        <p>|Developed with ❤️ by Ashwin Nair |      
    </div>
    """, unsafe_allow_html=True)
