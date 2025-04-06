import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests

# Example color variables — adjust to match your brand
PRIMARY_BG = "#F7F9FC"      # Light background color
SECONDARY_BG = "#FFFFFF"    # Another background or card color
TEXT_COLOR = "#333333"      # Standard text color
ACCENT_COLOR = "#FFA500"    # Example accent color (e.g., for highlights, buttons)
CHAT_USER_BG = "#E9F2FF"    # Light user bubble background
CHAT_ASSISTANT_BG = "#F2F2F2"  # Assistant bubble background

st.set_page_config(
    page_title="TravClan Navigator 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

# Inject custom CSS
def apply_custom_css():
    st.markdown(f"""
    <style>
        /* Overall app background */
        .stApp {{
            background-color: {PRIMARY_BG};
            color: {TEXT_COLOR};
        }}
        /* Center the main container and give it a max-width if you prefer */
        .main .block-container {{
            max-width: 1200px;
            background-color: {SECONDARY_BG};
            padding: 2rem 3rem;
            border-radius: 6px;
        }}
        /* Chat message bubbles */
        .stChatMessage-user {{
            background-color: {CHAT_USER_BG} !important;
            color: {TEXT_COLOR} !important;
        }}
        .stChatMessage-assistant {{
            background-color: {CHAT_ASSISTANT_BG} !important;
            color: {TEXT_COLOR} !important;
        }}
        /* Example accent color for headings or links */
        h1, h2, h3, h4, h5, h6 {{
            color: {ACCENT_COLOR};
        }}
        /* Optional: style the chat input container */
        .stChatInput {{
            background-color: {SECONDARY_BG};
            border-top: 1px solid #ccc;
        }}
    </style>
    """, unsafe_allow_html=True)

# The rest of your code remains largely the same.
from openai import OpenAI
client = None  # We'll init after secrets are loaded

def pdf_file_to_text(pdf_file):
    """Extract text from PDF."""
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def upload_and_index_file(pdf_file_path):
    """Uploads and indexes the PDF document into an OpenAI vector store."""
    with open(pdf_file_path, "rb") as file_stream:
        vector_store = client.vector_stores.create(name="TravClan Navigator Documents")
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file_stream]
        )
    return vector_store

def create_assistant_with_vector_store(vector_store):
    """Creates an assistant that uses the vector store for context."""
    assistant = client.beta.assistants.create(
        name="TravClan Navigator Assistant",
        instructions=(
            "Answer travel-related questions as precisely as possible using the provided context. "
            "If the answer is not fully contained in the internal documents, say 'answer not available in context'."
        ),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {"vector_store_ids": [vector_store.id]}
        }
    )
    return assistant

def generate_clarifying_question(user_question):
    prompt = (
        f"You are a travel expert. The user asked:\n\n"
        f"\"{user_question}\"\n\n"
        "What is a single, concise clarifying question you should ask to gather more travel details? "
        "Return only the question."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_answer(assistant_id, conversation_history, user_question):
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_question})
    
    thread = client.beta.threads.create(messages=messages)
    answer = ""
    start_time = time.time()
    
    with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
        for event in stream:
            if event.event == 'thread.message.delta':
                for delta_block in event.data.delta.content:
                    if delta_block.type == 'text':
                        answer += delta_block.text.value
    end_time = time.time()
    
    if "answer not available in context" in answer.lower():
        clarifying_question = generate_clarifying_question(user_question)
        answer = clarifying_question

    return answer

def main():
    global client
    # Retrieve API key from Streamlit secrets
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    # Initialize the OpenAI client with beta headers for assistants and vector stores.
    client = OpenAI(
        api_key=openai_api_key,
        default_headers={"OpenAI-Beta": "assistants=v2"}
    )

    # Apply custom CSS for theming
    apply_custom_css()

    st.title("TravClan Navigator 🌍🧭")
    st.write("Welcome! Ask anything about your trip or our internal processes. We'll consult internal docs and ask clarifying questions as needed.")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    if "vector_store" not in st.session_state:
        with st.spinner("Indexing travel documents..."):
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            st.session_state.vector_store = vector_store
            assistant = create_assistant_with_vector_store(vector_store)
            st.session_state.assistant = assistant
    else:
        assistant = st.session_state.assistant
    
    # Display existing conversation with chat UI
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    user_question = st.chat_input("Type your travel question here...")
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.spinner("Processing your query..."):
            answer = generate_answer(assistant.id, st.session_state.conversation_history, user_question)
        
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()
