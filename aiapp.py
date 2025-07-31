import streamlit as st
from typing import Generator
from groq import Groq
import fitz  # PyMuPDF for PDF
import pandas as pd
import docx
import pytesseract # Although imported, not used in the provided logic for OCR.
from PIL import Image # Although imported, not used in the provided logic for image processing.
import plotly.express as px
import time  # To measure response time
import streamlit_shadcn_ui as ui  # Streamlit Shadcn UI for buttons, etc.

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_icon="🏎️",
    layout="wide",
    page_title="Ashwin's AI Playground"
)

# -----------------------------------------------------------------------------
# ENHANCED STYLING FOR MODERN UI/UX
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* CSS Variables for easier theming */
    :root {
        --primary-bg: #0F0F0F; /* Darkest background for main content */
        --secondary-bg: #1A1A1A; /* Slightly lighter dark for sidebar/cards */
        --text-color: #F0F0F0; /* Light text for readability */
        --accent-color-user: #0A6EBD; /* Deep Blue for user messages */
        --accent-color-assistant: #263238; /* Dark Slate Gray for assistant messages */
        --border-color: #333333; /* Subtle border color */
        --shadow-color: rgba(0, 0, 0, 0.4); /* Stronger shadow for depth */
        --code-bg: rgba(0, 0, 0, 0.5); /* Dark background for inline code */
        --code-text: #A0C0E0; /* Light blue for code text */
    }

    /* General Page Styling */
    body, .main, .stApp {
        background-color: var(--primary-bg) !important;
        color: var(--text-color);
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; /* Modern font stack */
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    /* Scrollbar Styling */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: var(--secondary-bg);
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb {
        background: #555;
        border-radius: 5px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #777;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
        border-right: 1px solid var(--border-color);
    }
    [data-testid="stSidebar"] .stButton > button,
    [data-testid="stSidebar"] .stSelectbox > div > div,
    [data-testid="stSidebar"] .stSlider > div > div,
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background-color: #333; /* Darker background for controls */
        color: var(--text-color);
        border-color: var(--border-color);
    }
    [data-testid="stSidebar"] .stExpander > div > div {
        color: var(--text-color); /* Ensure expander titles are readable */
    }
    /* Expander icon color */
    [data-testid="stExpanderChevron"] svg {
        fill: var(--text-color);
    }


    /* Header Styling */
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .stMarkdown h1 {
        color: #FFFFFF !important; /* Ensure the main title is white */
    }

    /* Chat Container Styling - A modern chat look */
    .stChatMessage {
        padding: 1.2rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 12px var(--shadow-color); /* More prominent shadow */
        max-width: 75%; /* Slightly reduced width for better aesthetics */
        border: 1px solid var(--border-color);
        transition: all 0.2s ease-in-out; /* Smooth transition on hover */
    }
    .stChatMessage:hover {
        transform: translateY(-2px); /* Subtle lift on hover */
        box-shadow: 0 6px 15px var(--shadow-color);
    }

    /* User Message Styling - Aligned to the right */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-user"]) {
        background-color: var(--accent-color-user);
        margin-left: auto;
        border-bottom-right-radius: 0.25rem; /* Make user message slightly different */
        border-top-right-radius: 1.5rem;
    }

    /* Assistant Message Styling - Aligned to the left */
    div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageContent-assistant"]) {
        background-color: var(--accent-color-assistant);
        margin-right: auto;
        border-bottom-left-radius: 0.25rem; /* Make assistant message slightly different */
        border-top-left-radius: 1.5rem;
    }

    /* Markdown styling within chat messages for better readability */
    .stChatMessage p, .stChatMessage li, .stChatMessage code, .stChatMessage ol, .stChatMessage ul {
        font-size: 1.05rem;
        line-height: 1.6;
        color: var(--text-color);
    }
    .stChatMessage code { /* Inline code */
        background-color: var(--code-bg);
        padding: 0.2rem 0.4rem;
        border-radius: 5px;
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, Courier, monospace;
        color: var(--code-text);
    }
    .stChatMessage pre { /* Code blocks */
        background-color: var(--code-bg);
        border-radius: 8px;
        padding: 1rem;
        overflow-x: auto; /* Enable horizontal scrolling for long code lines */
        border: 1px solid rgba(255,255,255,0.1); /* Subtle border for code blocks */
    }
    .stChatMessage pre code {
        background: none; /* Override inline code background for blocks */
        padding: 0;
        font-size: 0.95rem;
        color: var(--code-text);
    }

    /* Chat Input Styling */
    div.stChatInputContainer {
        background-color: var(--primary-bg);
        border-top: 1px solid var(--border-color);
        padding: 1rem 0;
    }
    .stChatInput > div > div > input {
        background-color: #222; /* Darker input background */
        color: var(--text-color);
        border: 1px solid var(--border-color);
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
    }
    .stChatInput > div > div > button { /* Send button */
        background-color: var(--accent-color-user);
        color: white;
        border-radius: 0.75rem;
        padding: 0.75rem 1rem;
        transition: background-color 0.2s ease-in-out;
    }
    .stChatInput > div > div > button:hover {
        background-color: #0A5EAA; /* Darker shade on hover */
    }

    /* Streamlit Shadcn UI buttons (e.g., Clear Chat) */
    /* This targets a common generated class for Streamlit's custom components */
    .st-emotion-cache-k3wzrj button {
        background-color: var(--accent-color-user) !important;
        color: white !important;
        border: none !important;
        border-radius: 0.5rem !important;
        padding: 0.5rem 1rem !important;
        transition: background-color 0.2s ease-in-out;
    }
    .st-emotion-cache-k3wzrj button:hover {
        background-color: #0A5EAA !important; /* Darker shade on hover */
    }

    /* Metric elements (in sidebar) */
    [data-testid="stMetricValue"] {
        color: var(--accent-color-user); /* Make metric values stand out */
    }
    [data-testid="stMetricLabel"] {
        color: var(--text-color);
    }


    /* Footer Styling */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #000000;
        color: white;
        text-align: center;
        padding: 10px 0;
        z-index: 999;
        border-top: 1px solid var(--border-color);
    }
    </style>
    """,
    unsafe_allow_html=True
)

def icon_with_text(emoji: str, text: str):
    """Shows an emoji and text as a Notion-style page header."""
    st.write(
        f'<div style="display: flex; align-items: center; margin-bottom: 20px;">'
        f'<span style="font-size: 78px; line-height: 1; margin-right: 15px;">{emoji}</span>'
        f'<h1 style="font-size: 48px; line-height: 1.2;">{text}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

# Helper function to convert CSS variables to actual hex values for Plotly
# THIS FUNCTION IS MOVED HERE TO BE DEFINED BEFORE ITS USAGE
def var_to_hex(var_name):
    """
    Returns the hex value corresponding to a CSS variable name.
    This is a hardcoded mapping as Streamlit's Python runs on server
    and CSS on client. For static CSS variables defined above, this is fine.
    """
    mapping = {
        '--primary-bg': '#0F0F0F',
        '--secondary-bg': '#1A1A1A',
        '--text-color': '#F0F0F0',
        '--accent-color-user': '#0A6EBD',
        '--accent-color-assistant': '#263238',
        '--border-color': '#333333',
        '--shadow-color': 'rgba(0, 0, 0, 0.4)',
        '--code-bg': 'rgba(0, 0, 0, 0.5)',
        '--code-text': '#A0C0E0',
    }
    return mapping.get(var_name, '#FFFFFF') # Default to white if not found


# -----------------------------------------------------------------------------
# MAIN TITLE
# -----------------------------------------------------------------------------
icon_with_text("🏎️", "Ashwin's AI Playground")

# Initialize session state for api_key if not present
if "api_key" not in st.session_state:
    st.session_state.api_key = "" # Initialize as empty string

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION - REORGANIZED FOR BETTER UX
# -----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛠️ Controls & Options")
    st.write("Configure the model, provide context, and manage your chat session.")

    # --- API Key Input ---
    with st.expander("🔑 API Credentials", expanded=True):
        # Always update session state with the current value of the text input
        api_key_input = st.text_input(
            "Enter your Groq API Key",
            type="password",
            key="api_key_input",
            value=st.session_state.api_key # Set the initial value from session state
        )
        # Update session state with the actual value from the input field
        st.session_state.api_key = api_key_input

        if st.session_state.api_key:
            st.success("API Key accepted!", icon="✅")
        else:
            st.warning("Please enter your Groq API Key to enable AI features.", icon="⚠️")


    # --- Model Selection ---
    # Sourced from https://console.groq.com/docs/models [1]
    models = {
        # Production Models
        "gemma2-9b-it": {"name": "Gemma 2 9B IT", "tokens": 8192, "developer": "Google", "type": "Text", "max_completion_tokens": 8192},
        "llama-3.1-8b-instant": {"name": "LLaMA 3.1 8B (Instant)", "tokens": 131072, "developer": "Meta", "type": "Text", "max_completion_tokens": 131072},
        "llama-3.3-70b-versatile": {"name": "LLaMA 3.3 70B (Versatile)", "tokens": 131072, "developer": "Meta", "type": "Text", "max_completion_tokens": 32768},
        "meta-llama/llama-guard-4-12b": {"name": "LLaMA Guard 4 12B", "tokens": 131072, "developer": "Meta", "type": "Text", "max_completion_tokens": 1024},
        "whisper-large-v3": {"name": "Whisper Large V3", "tokens": None, "developer": "OpenAI", "type": "File (Audio)", "max_completion_tokens": None},
        "whisper-large-v3-turbo": {"name": "Whisper Large V3 Turbo", "tokens": None, "developer": "OpenAI", "type": "File (Audio)", "max_completion_tokens": None},
        # Preview Models
        "deepseek-r1-distill-llama-70b": {"name": "DeepSeek-R1 Distill Llama 70B (Preview)", "tokens": 131072, "developer": "DeepSeek / Meta", "type": "Text", "max_completion_tokens": 131072},
        "meta-llama/llama-4-maverick-17b-128e-instruct": {"name": "LLaMA 4 Maverick 17B Instruct (Preview)", "tokens": 131072, "developer": "Meta", "type": "Text", "max_completion_tokens": 8192},
        "meta-llama/llama-4-scout-17b-16e-instruct": {"name": "LLaMA 4 Scout 17B Instruct (Preview)", "tokens": 131072, "developer": "Meta", "type": "Text", "max_completion_tokens": 8192},
        "meta-llama/llama-prompt-guard-2-22m": {"name": "LLaMA Prompt Guard 2 22M (Preview)", "tokens": 512, "developer": "Meta", "type": "Text", "max_completion_tokens": 512},
        "meta-llama/llama-prompt-guard-2-86m": {"name": "LLaMA Prompt Guard 2 86M (Preview)", "tokens": 512, "developer": "Meta", "type": "Text", "max_completion_tokens": 512},
        "moonshotai/kimi-k2-instruct": {"name": "Kimi K2 Instruct (Preview)", "tokens": 131072, "developer": "Moonshot AI", "type": "Text", "max_completion_tokens": 16384},
        "playai-tts": {"name": "PlayAI TTS (Preview)", "tokens": 8192, "developer": "PlayAI", "type": "Text", "max_completion_tokens": 8192},
        "playai-tts-arabic": {"name": "PlayAI TTS Arabic (Preview)", "tokens": 8192, "developer": "PlayAI", "type": "Text", "max_completion_tokens": 8192},
        "qwen/qwen3-32b": {"name": "Qwen 3 32B (Preview)", "tokens": 131072, "developer": "Alibaba Cloud", "type": "Text", "max_completion_tokens": 40960},
        # Original models from the script
        "llama3-70b-8192": {"name": "Meta LLaMA 3 70B", "tokens": 8192, "developer": "Meta", "type": "Text", "max_completion_tokens": 8192},
        "llama3-8b-8192": {"name": "Meta LLaMA 3 8B", "tokens": 8192, "developer": "Meta", "type": "Text", "max_completion_tokens": 8192},
        "mixtral-8x7b-32768": {"name": "Mixtral 8x7B 32768", "tokens": 32768, "developer": "Mistral", "type": "Text", "max_completion_tokens": 32768},
    }

    with st.expander("⚙️ Model & Parameters", expanded=True):
        if "selected_model" not in st.session_state:
            st.session_state.selected_model = None

        default_model_key = "llama-3.1-8b-instant"
        model_option = st.selectbox(
            "Choose a model:",
            options=list(models.keys()),
            format_func=lambda x: models[x]["name"],
            index=list(models.keys()).index(default_model_key)
        )

        # If model changes, clear messages and rerun
        if st.session_state.selected_model != model_option:
            st.session_state.messages = []
            st.session_state.selected_model = model_option
            st.rerun() # Rerun to ensure chat history is cleared and prompt reflects new model

        max_tokens_range = models[model_option].get("max_completion_tokens", 8192)
        max_tokens = st.slider(
            "Max Tokens:", min_value=512, max_value=max_tokens_range,
            value=min(8192, max_tokens_range), step=512,
            help=f"Set the max tokens for the response. Max for this model: {max_tokens_range}"
        )

    # --- AGENTS DICTIONARY & CONTEXT ---
    agents = {
        "None (No specialized agent)": {"name": "None (General Purpose)", "system_prompt": ""},
        "excel_basics_tutor": {"name": "Excel Basics Tutor", "system_prompt": "You are an 'Excel Basics Tutor' for absolute beginners. Explain concepts clearly, provide simple examples, and be patient. Focus on fundamental operations, formulas, and data organization. Always start with a friendly greeting and end with an offer for more help."},
        "python_developer": {"name": "Python Developer Assistant", "system_prompt": "You are a Python Developer Assistant. Provide accurate, efficient, and well-commented Python code solutions. Explain complex concepts clearly and suggest best practices. Ask clarifying questions if needed."},
        "creative_writer": {"name": "Creative Writer Muse", "system_prompt": "You are a Creative Writer Muse. Help users brainstorm ideas, develop plots, create characters, and refine their prose. Be imaginative, encouraging, and provide constructive feedback. Focus on storytelling, world-building, and literary techniques."},
        "technical_documentation_specialist": {"name": "Technical Documentation Specialist", "system_prompt": "You are a Technical Documentation Specialist. Your goal is to provide clear, concise, and accurate documentation. Use markdown formatting effectively, including code blocks, lists, and headings, to make information easy to digest. Explain complex technical topics in simple terms. Adopt a professional and systematic tone."},
    }

    with st.expander("📜 Agent & Context", expanded=False):
        agent_key = st.selectbox(
            "Choose an agent persona:",
            options=list(agents.keys()),
            format_func=lambda x: agents[x]["name"]
        )
        selected_agent_prompt = agents[agent_key]["system_prompt"]

        uploaded_file = st.file_uploader(
            "Upload a file for context (PDF, DOCX, XLSX)",
            type=["pdf", "docx", "xlsx"]
        )

    # --- Chat Controls ---
    with st.expander("💬 Chat Controls", expanded=False):
        search_query = st.text_input("Search chat history:")
        col1, col2 = st.columns(2)
        with col1:
            if ui.button(text="Clear Chat", key="clear_chat_btn", className="w-full"):
                st.session_state.messages = []
                st.session_state.knowledge_base = "" # Also clear knowledge base on chat clear
                st.session_state.last_uploaded = None
                st.session_state.response_times = [] # Clear response times too
                st.session_state.chat_count = 0
                st.session_state.agent_usage = {}
                st.rerun()
        with col2:
            # Placeholder for potential future feature or instruction
            st.info("Scroll to the bottom for latest messages.")


    # --- SESSION STATS ---
    with st.sidebar.expander("📊 Session Dashboard", expanded=True):
        if "chat_count" not in st.session_state:
            st.session_state.chat_count = 0
        if "agent_usage" not in st.session_state:
            st.session_state.agent_usage = {k: 0 for k in agents.keys()} # Initialize all agents to 0
        if agent_key not in st.session_state.agent_usage: # Ensure newly added agents are initialized
            st.session_state.agent_usage[agent_key] = 0
        if "response_times" not in st.session_state:
            st.session_state.response_times = []

        st.metric("Total Queries in Session", st.session_state.chat_count)
        if st.session_state.response_times:
            avg_response = sum(st.session_state.response_times) / len(st.session_state.response_times)
            st.metric("Avg. Response Time (s)", f"{avg_response:.2f}")
        else:
            st.metric("Avg. Response Time (s)", "N/A")


        st.write("**Agent Usage:**")
        # Ensure only agents that have been used or are present in the current session are shown
        current_agent_usage = {k: v for k, v in st.session_state.agent_usage.items() if v > 0 or k == agent_key}

        if current_agent_usage:
            agent_df = pd.DataFrame(
                current_agent_usage.items(),
                columns=['Agent Key', 'Count']
            ).sort_values('Count', ascending=False)
            agent_df['Agent Name'] = agent_df['Agent Key'].apply(lambda x: agents.get(x, {'name': x})['name'])
            st.dataframe(agent_df[['Agent Name', 'Count']], use_container_width=True, hide_index=True)
        else:
            st.info("No agent usage data yet.")

        if st.session_state.response_times:
            fig = px.histogram(st.session_state.response_times, nbins=10, title="Response Time Distribution (s)")
            fig.update_layout(showlegend=False, yaxis_title="Count", xaxis_title="Response Time (s)",
                              plot_bgcolor=var_to_hex('--secondary-bg'), # Match plot background to sidebar
                              paper_bgcolor=var_to_hex('--secondary-bg'),
                              font_color=var_to_hex('--text-color'))
            st.plotly_chart(fig, use_container_width=True)


# -----------------------------------------------------------------------------
# FILE PROCESSING & KNOWLEDGE BASE
# -----------------------------------------------------------------------------
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return ""

def extract_text_from_word(file):
    try:
        doc_obj = docx.Document(file)
        return "\n".join(para.text for para in doc_obj.paragraphs)
    except Exception as e:
        st.error(f"Error reading DOCX: {e}")
        return ""

def extract_text_from_excel(file):
    try:
        df = pd.read_excel(file, engine='openpyxl')
        return df.to_string()
    except Exception as e:
        st.error(f"Error reading XLSX: {e}")
        return ""

knowledge_base = ""
if uploaded_file:
    # Check if a new file was uploaded or if it's the same file
    if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
        with st.spinner(f"Extracting text from {uploaded_file.name}..."):
            file_type = uploaded_file.type
            if file_type == "application/pdf":
                knowledge_base = extract_text_from_pdf(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                knowledge_base = extract_text_from_word(uploaded_file)
            elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                knowledge_base = extract_text_from_excel(uploaded_file)
            else:
                st.warning("Unsupported file type. Please upload PDF, DOCX, or XLSX.")
                knowledge_base = ""

            if knowledge_base:
                st.session_state.knowledge_base = knowledge_base
                st.session_state.last_uploaded = uploaded_file.name
                st.toast(f"✅ Successfully processed {uploaded_file.name}!", icon="📄")
            else:
                st.error(f"Failed to extract text from {uploaded_file.name}. See error above.")
                st.session_state.knowledge_base = ""
                st.session_state.last_uploaded = None
    else:
        # If the same file is re-selected, use the cached content
        knowledge_base = st.session_state.get("knowledge_base", "")

# Display extracted knowledge base content if available
if "knowledge_base" in st.session_state and st.session_state.knowledge_base:
    with st.expander("📄 View Content from Uploaded File", expanded=False):
        st.text_area("Extracted Text", st.session_state.knowledge_base, height=200, disabled=True)
elif "last_uploaded" in st.session_state and st.session_state.last_uploaded:
    st.info(f"File '{st.session_state.last_uploaded}' processed successfully, but no content available in session. Clear chat to re-upload.")


# -----------------------------------------------------------------------------
# INITIALIZE GROQ CLIENT - REVISED LOGIC
# -----------------------------------------------------------------------------
# Initialize client and last_groq_api_key in session state if they don't exist
if "client" not in st.session_state:
    st.session_state.client = None
if "last_groq_api_key" not in st.session_state:
    st.session_state.last_groq_api_key = None

# If an API key is provided and it's different from the one used last time
if st.session_state.api_key and st.session_state.api_key != st.session_state.last_groq_api_key:
    try:
        st.session_state.client = Groq(api_key=st.session_state.api_key)
        st.session_state.last_groq_api_key = st.session_state.api_key # Store the key that successfully initialized the client
        # st.toast("Groq client initialized successfully!") # Can be enabled for more feedback
    except Exception as e:
        st.error(f"Failed to initialize Groq client: {e}. Please check your API key and network connection.")
        st.session_state.client = None
        st.session_state.last_groq_api_key = None # Clear the stored key if initialization failed
elif not st.session_state.api_key: # If API key is empty or cleared
    st.session_state.client = None
    st.session_state.last_groq_api_key = None


# -----------------------------------------------------------------------------
# DISPLAY CHAT MESSAGES
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Filter messages based on search query
display_messages = st.session_state.messages
if search_query:
    filtered_messages = [
        msg for msg in st.session_state.messages if search_query.lower() in msg["content"].lower()
    ]
    if not filtered_messages:
        st.info(f"No messages found containing '{search_query}'.")
    display_messages = filtered_messages

for message in display_messages:
    avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'
    with st.chat_message(message["role"], avatar=avatar):
         st.markdown(message["content"], unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CHAT INPUT AND RESPONSE LOGIC
# -----------------------------------------------------------------------------
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Generator to yield chunks of content from chat completion."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

def handle_chat_input(prompt):
    """Handles user input, calls the LLM, and updates chat history."""
    
    # Ensure client is initialized before making API call
    if st.session_state.client is None:
        st.error("Groq client not initialized. Please enter your API key in the sidebar.")
        return

    st.session_state.chat_count += 1
    st.session_state.agent_usage[agent_key] = st.session_state.agent_usage.get(agent_key, 0) + 1

    # Append user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Construct message list for API, including system prompt and knowledge base
    conversation_messages = []
    if selected_agent_prompt.strip():
        conversation_messages.append({"role": "system", "content": selected_agent_prompt})
    
    if "knowledge_base" in st.session_state and st.session_state.knowledge_base.strip():
        kb_content = f"--- Contextual Information ---\n{st.session_state.knowledge_base}\n---\n"
        conversation_messages.append({"role": "system", "content": kb_content})

    # Add previous chat messages to conversation
    conversation_messages.extend(st.session_state.messages)

    try:
        start_time = time.time()
        
        chat_completion = st.session_state.client.chat.completions.create(
            model=model_option,
            messages=conversation_messages,
            max_tokens=max_tokens,
            stream=True
        )

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in generate_chat_responses(chat_completion):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌", unsafe_allow_html=True) # Add blinking cursor
            response_placeholder.markdown(full_response, unsafe_allow_html=True) # Final display without cursor

        response_time = time.time() - start_time
        st.session_state.response_times.append(response_time)

        # Append assistant response to session state
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    except Exception as e:
        st.error(f"An error occurred while generating response: {e}", icon="🚨")
    
    # Rerun to display the new messages immediately
    st.rerun()


# --- Main app logic ---
if not st.session_state.api_key:
    st.info("Please enter your Groq API key in the sidebar to begin.")
else:
    # Only show chat input if API key is present
    if prompt := st.chat_input("Ask me anything..."):
        handle_chat_input(prompt)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer">
        <p>| Developed with ❤️ by Ashwin Nair |</p>
    </div>
    """,
    unsafe_allow_html=True
)
