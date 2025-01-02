import streamlit as st
from typing import Generator
from groq import Groq
import fitz  # PyMuPDF for PDF
import pandas as pd
import docx
# Removed: from moviepy.editor import VideoFileClip
import pytesseract
from PIL import Image
import plotly.express as px
import time  # To measure response time
import streamlit_shadcn_ui as ui  # Streamlit Shadcn UI for buttons, etc.

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_icon="💬",
    layout="wide",
    page_title="Ashwin's LLM - Enhanced"
)

# -----------------------------------------------------------------------------
# ADD STYLE FOR BETTER READABILITY & UNIFIED BACKGROUND
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Unify the main page background color */
    body, .main, .block-container {
        background-color: #0F0F0F !important;
    }

    /* Chat message bubble for overall styling */
    .stChatMessage {
        background-color: #0F0F0F !important;
        border-radius: 10px;
        margin: 10px 0;
        padding: 25px;
        font-size: 1.2rem;
        line-height: 3;
        color: #FEFCFC;
        box-shadow: 0 5px 5px rgba(0, 0, 0, 0.05);
    }

    /* A different background for user messages */
    .stChatMessageUser {
        background-color: #0F0F0F !important; 
    }

    /* Assistant messages remain white for subtle contrast */
    .stChatMessageAssistant {
        background-color: #0F0F0F !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

def icon_with_text(emoji: str, text: str):
    """Shows an emoji and text as a Notion-style page header."""
    st.write(
        f'<div style="display: flex; align-items: center;">'
        f'<span style="font-size: 78px; line-height: 1; margin-right: 15px;">{emoji}</span>'
        f'<h1 style="font-size: 48px; line-height: 1.2;">{text}</h1>'
        f'</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# MAIN TITLE
# -----------------------------------------------------------------------------
icon_with_text("🏎️", "Ashwin's AI Playground")

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("Options")
st.sidebar.header("Ashwin's AI Playground")
st.sidebar.write("Experiment with different AI models, functionalities, and extended features.")

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

# -----------------------------------------------------------------------------
# AGENTS DICTIONARY
# -----------------------------------------------------------------------------
agents = {
    "None (No specialized agent)": {
        "name": "None (No specialized agent)",
        "system_prompt": ""
    },
    "excel_basics_tutor": {
        "name": "Excel Basics Tutor",
        "system_prompt": (
            "You are an 'Excel Basics Tutor' responsible for coaching absolute beginners in Microsoft Excel. "
            "Your teaching emphasizes correct data entry, formatting fundamentals (numbers, dates, text), "
            "and an introduction to essential formulas like SUM, AVERAGE, COUNT, and basic IF statements.\n\n"
            "When providing guidance, maintain a patient, step-by-step approach. Offer screenshots or bullet points "
            "to illustrate core concepts such as creating new sheets, adjusting column widths, and using the 'Fill Handle' "
            "effectively. In your examples, favor small datasets (5–10 rows) to help learners build confidence "
            "before taking on larger tasks.\n\n"
            "Your tone should be encouraging, informative, and empathetic to the frustrations of new learners. "
            "Always confirm understanding by reiterating key steps or suggesting quick exercises they can do. "
            "Use everyday examples (budget tracking, to-do lists) to enhance relevance."
        )
    },
    # ... [Remaining agents are unchanged]
    # (For brevity, omitted other agents. Ensure all agents are included as in the original code.)
}

# -----------------------------------------------------------------------------
# SELECT AN AGENT
# -----------------------------------------------------------------------------
st.sidebar.subheader("Select an Agent")
agent_key = st.sidebar.selectbox(
    "Choose an agent:",
    options=list(agents.keys()),
    format_func=lambda x: agents[x]["name"]
)
selected_agent_prompt = agents[agent_key]["system_prompt"]

# -----------------------------------------------------------------------------
# SIDEBAR FOR TOPIC SELECTION (DEMO ONLY)
# -----------------------------------------------------------------------------
st.sidebar.subheader("Select a Topic")
topic = st.sidebar.selectbox(
    "Choose a topic",
    ["General", "Data Analysis", "Machine Learning", "Natural Language Processing"]
)

# -----------------------------------------------------------------------------
# FILE UPLOAD
# -----------------------------------------------------------------------------
st.sidebar.subheader("Upload Files")
uploaded_file = st.sidebar.file_uploader(
    "Upload a PDF, Word, or Excel file",
    type=["pdf", "docx", "xlsx"]  # Removed video file types
)

# Define extraction functions
def extract_text_from_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_word(file):
    doc_obj = docx.Document(file)
    text = ""
    for para in doc_obj.paragraphs:
        text += para.text + "\n"
    return text

def extract_text_from_excel(file):
    df = pd.read_excel(file)
    return df.to_string()

# Removed: extract_text_from_video function

def extract_text_from_uploaded_file(file):
    if file.type == "application/pdf":
        return extract_text_from_pdf(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return extract_text_from_word(file)
    elif file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return extract_text_from_excel(file)
    # Removed video file handling
    else:
        return "Unsupported file type."

# Initialize knowledge_base
knowledge_base = ""
if uploaded_file:
    knowledge_base = extract_text_from_uploaded_file(uploaded_file)
    st.write("**Extracted text from uploaded file:**")
    st.write(knowledge_base)

# -----------------------------------------------------------------------------
# CLEAR CHAT & SCROLL BUTTONS
# -----------------------------------------------------------------------------
if st.sidebar.button("Scroll to top"):
    st.experimental_set_query_params()  # This helps reset the view in some Streamlit versions
    st.rerun()

if st.sidebar.button("Clear Chat"):
    if "messages" in st.session_state:
        st.session_state.messages = []
    st.rerun()

# -----------------------------------------------------------------------------
# MODEL SELECTION
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

models = {
    "llama3-groq-70b-8192-tool-use-preview": {
        "name": "LLaMA3 Groq 70B Tool Use (Preview)",
        "tokens": 8192,
        "developer": "Groq",
        "type": "Text"
    },
    "llama3-groq-8b-8192-tool-use-preview": {
        "name": "LLaMA3 Groq 8B Tool Use (Preview)",
        "tokens": 8192,
        "developer": "Groq",
        "type": "Text"
    },
    "llama-3.3-70b-specdec": {
        "name": "LLaMA 3.3 70B (SpecDec)",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.1-70b-specdec": {
        "name": "LLaMA 3.1 70B (SpecDec)",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-1b-preview": {
        "name": "LLaMA 3.2 1B (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-3b-preview": {
        "name": "LLaMA 3.2 3B (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.2-11b-vision-preview": {
        "name": "LLaMA 3.2 11B Vision (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text and Vision"
    },
    "llama-3.2-90b-vision-preview": {
        "name": "LLaMA 3.2 90B Vision (Preview)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text and Vision"
    },
    "distil-whisper-large-v3-en": {
        "name": "Distil-Whisper Large V3 (English)",
        "tokens": None,
        "developer": "HuggingFace",
        "type": "File (Audio)"
    },
    "gemma2-9b-it": {
        "name": "Gemma 2 9B",
        "tokens": 8192,
        "developer": "Google",
        "type": "Text"
    },
    "gemma-7b-it": {
        "name": "Gemma 7B (DEPRECATED)",
        "tokens": 8192,
        "developer": "Google",
        "type": "Text"
    },
    # Ensure context window is 128k and max output tokens is 32768 for llama-3.3-70b-versatile:
    "llama-3.3-70b-versatile": {
        "name": "LLaMA 3.3 70B (Versatile)",
        "tokens": 131072,  # 128k context window
        "developer": "Meta",
        "type": "Text",
        "max_output_tokens": 32768  # Enforced maximum output tokens
    },
    "llama-3.1-70b-versatile": {
        "name": "LLaMA 3.1 70B (Versatile) (DEPRECATED)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-3.1-8b-instant": {
        "name": "LLaMA 3.1 8B (Instant)",
        "tokens": 131072,  # 128k
        "developer": "Meta",
        "type": "Text"
    },
    "llama-guard-3-8b": {
        "name": "LLaMA Guard 3 8B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama3-70b-8192": {
        "name": "Meta LLaMA 3 70B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "llama3-8b-8192": {
        "name": "Meta LLaMA 3 8B",
        "tokens": 8192,
        "developer": "Meta",
        "type": "Text"
    },
    "mixtral-8x7b-32768": {
        "name": "Mixtral 8x7B 32768",
        "tokens": 32768,
        "developer": "Mistral",
        "type": "Text"
    },
    "whisper-large-v3": {
        "name": "Whisper Large V3",
        "tokens": None,
        "developer": "OpenAI",
        "type": "File (Audio)"
    },
    "whisper-large-v3-turbo": {
        "name": "Whisper Large V3 Turbo",
        "tokens": None,
        "developer": "OpenAI",
        "type": "File (Audio)"
    },
    "llava-v1.5-7b-4096-preview": {
        "name": "LLaVA 1.5 7B (Preview)",
        "tokens": 4096,
        "developer": "Haotian Liu",
        "type": "Text"
    }
}

default_model_key = "llama-3.1-8b-instant" if "llama-3.1-8b-instant" in models else list(models.keys())[0]
model_option = st.sidebar.selectbox(
    "Choose a model:",
    options=list(models.keys()),
    format_func=lambda x: models[x]["name"],
    index=list(models.keys()).index(default_model_key) if default_model_key in models else 0
)

# Detect model change and clear chat history if model has changed
if st.session_state.selected_model != model_option:
    st.session_state.messages = []
    st.session_state.selected_model = model_option

# Determine the maximum tokens range:
# Priority: "max_output_tokens" if it exists, else fallback to "tokens".
if models[model_option].get("max_output_tokens"):
    max_tokens_range = models[model_option]["max_output_tokens"]
else:
    max_tokens_range = models[model_option]["tokens"] if isinstance(models[model_option]["tokens"], int) else 1024

st.sidebar.subheader("Max Tokens")
max_tokens = st.sidebar.slider(
    "Max Tokens:",
    min_value=512,
    max_value=max_tokens_range if isinstance(max_tokens_range, int) and max_tokens_range > 512 else 512,
    value=min(7680, max_tokens_range) if isinstance(max_tokens_range, int) and max_tokens_range > 7680 else 512,
    step=512,
    help=(
        "Adjust the maximum number of tokens for the model's response. "
        f"Max for this model: {max_tokens_range}"
    )
)

# -----------------------------------------------------------------------------
# INITIALIZE GROQ CLIENT
# -----------------------------------------------------------------------------
if "client" not in st.session_state and st.session_state.api_key:
    st.session_state.client = Groq(api_key=st.session_state.api_key)

# -----------------------------------------------------------------------------
# SEARCHABLE CHAT HISTORY
# -----------------------------------------------------------------------------
st.sidebar.subheader("Chat History Search")
search_query = st.sidebar.text_input("Search messages by keyword:")
search_results = []

# If user enters a search query, filter chat history
if search_query and "messages" in st.session_state:
    for idx, msg in enumerate(st.session_state.messages):
        if search_query.lower() in msg["content"].lower():
            search_results.append((idx, msg["role"], msg["content"]))

if search_results:
    st.sidebar.write("**Search Results:**")
    for res in search_results:
        index, role, content = res
        st.sidebar.write(f"- **Message #{index}** ({role}): {content[:50]}...")
    st.sidebar.write("**Note**: Click 'Jump to Latest' below or scroll in the main window to view details.")

# -----------------------------------------------------------------------------
# PROMPT TEMPLATES & QUICK ACTIONS
# -----------------------------------------------------------------------------
st.sidebar.subheader("Prompt Templates & Quick Actions")
quick_prompts = {
    "Explain an Excel formula": "Could you explain how to use the SUMIF function in Excel?",
    "Show me a WFM Forecast Example": "Provide a sample workforce management forecast scenario for a call center.",
    "Demo Python Code": "Show me a simple Python snippet for data analysis in a call center."
}

selected_quick_prompt = st.sidebar.selectbox(
    "Select a Quick Prompt:",
    options=["None"] + list(quick_prompts.keys())
)

if selected_quick_prompt != "None":
    if "auto_prompt_triggered" not in st.session_state:
        st.session_state.auto_prompt_triggered = False

    if not st.session_state.auto_prompt_triggered:
        st.session_state.auto_prompt_triggered = True
        st.experimental_set_query_params()  # For refreshing the UI
        # Immediately add the chosen quick prompt to messages and rerun
        st.session_state.messages.append({"role": "user", "content": quick_prompts[selected_quick_prompt]})
        st.rerun()

# -----------------------------------------------------------------------------
# CONSOLIDATED METRICS & ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------
if "chat_count" not in st.session_state:
    st.session_state.chat_count = 0
if "agent_usage" not in st.session_state:
    st.session_state.agent_usage = {}
if agent_key not in st.session_state.agent_usage:
    st.session_state.agent_usage[agent_key] = 0
if "total_time_in_training" not in st.session_state:
    st.session_state.total_time_in_training = 0.0
if "response_times" not in st.session_state:
    st.session_state.response_times = []

# Display in a collapsible form
with st.sidebar.expander("Session Stats / Usage Dashboard"):
    st.write(f"**Number of Queries Asked:** {st.session_state.chat_count}")
    st.write("**Most-used Agent Roles:**")
    sorted_agent_usage = sorted(st.session_state.agent_usage.items(), key=lambda x: x[1], reverse=True)
    for a, count in sorted_agent_usage:
        st.write(f"- {agents[a]['name']}: {count} times")

    st.write(f"**Total Time in Training (seconds):** {round(st.session_state.total_time_in_training,2)}")

    if st.session_state.response_times:
        avg_response = sum(st.session_state.response_times) / len(st.session_state.response_times)
        st.write(f"**Average Chat Response Time (seconds):** {round(avg_response,2)}")
        # Optional: distribution plot
        fig = px.histogram(st.session_state.response_times, nbins=10, title="Response Time Distribution")
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------------------------------------------------------
# DISPLAY CHAT MESSAGES (Auto-Scroll or Jump to Latest)
# -----------------------------------------------------------------------------
chat_container = st.container()
with chat_container:
    for i, message in enumerate(st.session_state.messages):
        role_class = "stChatMessageAssistant" if message["role"] == "assistant" else "stChatMessageUser"
        avatar = '🤖' if message["role"] == "assistant" else '👨‍💻'

        if "```" in message["content"]:
            # Inline code block rendering
            parts = message["content"].split("```")
            for idx, part in enumerate(parts):
                if idx % 2 == 0:
                    # Non-code text
                    if part.strip():
                        with st.chat_message(message["role"], avatar=avatar):
                            st.markdown(f"<div class='stChatMessage {role_class}'>{part}</div>", unsafe_allow_html=True)
                else:
                    # Code block
                    lines = part.strip().split("\n")
                    lang = "text"
                    if lines and lines[0].startswith("python"):
                        lang = "python"
                        lines = lines[1:]
                    code_content = "\n".join(lines)
                    with st.chat_message(message["role"], avatar=avatar):
                        st.markdown(f"<div class='stChatMessage {role_class}'>", unsafe_allow_html=True)
                        st.code(code_content, language=lang)
                        st.markdown("</div>", unsafe_allow_html=True)
        else:
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(f"<div class='stChatMessage {role_class}'>{message['content']}</div>", unsafe_allow_html=True)

if st.button("Jump to Latest Message"):
    st.rerun()

# -----------------------------------------------------------------------------
# GENERATOR FOR PARTIAL RESPONSES
# -----------------------------------------------------------------------------
def generate_chat_responses(chat_completion) -> Generator[str, None, None]:
    """Yield chat response content from the Groq API response."""
    for chunk in chat_completion:
        if chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content

# -----------------------------------------------------------------------------
# MAIN FUNCTION TO HANDLE CHAT INPUT
# -----------------------------------------------------------------------------
def handle_chat_input(prompt):
    st.session_state.chat_count += 1  # Increment chat count
    st.session_state.agent_usage[agent_key] += 1  # Increment usage of current agent

    # User prompt in the session
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user's message
    with st.chat_message("user", avatar='👨‍💻'):
        st.markdown(f"<div class='stChatMessage stChatMessageUser'>{prompt}</div>", unsafe_allow_html=True)

    # Build conversation
    conversation_messages = []
    if selected_agent_prompt.strip():
        conversation_messages.append({"role": "system", "content": selected_agent_prompt})
    if knowledge_base.strip():
        conversation_messages.append({"role": "system", "content": knowledge_base})

    for m in st.session_state.messages:
        conversation_messages.append({"role": m["role"], "content": m["content"]})

    full_response = ""
    start_time = time.time()

    try:
        chat_completion = st.session_state.client.chat.completions.create(
            model=model_option,
            messages=conversation_messages,
            max_tokens=max_tokens,
            stream=True
        )
        # Stream the assistant's response
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"<div class='stChatMessage stChatMessageAssistant'>", unsafe_allow_html=True)
            for partial_text in generate_chat_responses(chat_completion):
                full_response += partial_text
                st.markdown(partial_text, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(e, icon="🚨")

    response_time = time.time() - start_time
    st.session_state.total_time_in_training += response_time
    st.session_state.response_times.append(response_time)

    if full_response:
        # Append a note about response time
        full_response += f"\n\n_Response generated in {response_time:.2f} seconds._"
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Show response time in sidebar
    st.sidebar.metric(label="Response Time (seconds)", value=f"{response_time:.2f}")

    # Rerun to refresh the UI
    st.rerun()

# -----------------------------------------------------------------------------
# CHAT INPUT BOX
# -----------------------------------------------------------------------------
prompt = st.chat_input("Enter your prompt here...")
if prompt:
    handle_chat_input(prompt)

# -----------------------------------------------------------------------------
# FOOTER
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div style="position: fixed; bottom: 0; width: 100%; text-align: center;
                padding: 10px; background-color: #000000;">
        <p style="color: white;">|Developed with ❤️ by Ashwin Nair |</p>
    </div>
    """,
    unsafe_allow_html=True
)
