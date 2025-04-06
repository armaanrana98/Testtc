import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests
import os

st.set_page_config(
    page_title="TravClan Navigator 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"
VECTOR_STORE_ID_FILE = "vector_store_id.txt"  # Local file for simulating persistence

# Retrieve API key from Streamlit secrets.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client with beta headers.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def apply_custom_css():
    """
    Apply custom CSS for a balanced, modern theme.
    """
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        .stApp {
            background: linear-gradient(180deg, #f9f9f9, #eaeaea);
            color: #333333;
            font-family: 'Roboto', sans-serif;
        }
        .main .block-container {
            max-width: 900px;
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        .stChatMessage-user {
            background-color: #d0e6ff !important;
            color: #0d1b2a !important;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        }
        .stChatMessage-assistant {
            background-color: #dfffd8 !important;
            color: #0d1b2a !important;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        }
        .stChatInput {
            background-color: #ffffff !important;
            border-top: 1px solid #cccccc;
            padding: 1rem;
        }
        .stChatInput textarea {
            background-color: #f4f4f4 !important;
            color: #333333 !important;
            border: 1px solid #cccccc !important;
            border-radius: 8px !important;
            font-family: 'Roboto', sans-serif;
            padding: 0.6rem !important;
            font-size: 1rem;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #0066cc;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

def pdf_file_to_text(pdf_file):
    """Extract text from a PDF using PyPDF2."""
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

def get_persistent_vector_store():
    """
    Checks if a vector store ID exists in a local file.
    If yes, retrieves the vector store; if not, creates a new one and saves the ID.
    """
    if os.path.exists(VECTOR_STORE_ID_FILE):
        with open(VECTOR_STORE_ID_FILE, "r") as f:
            vector_store_id = f.read().strip()
        try:
            vector_store = client.vector_stores.retrieve(vector_store_id)
            return vector_store
        except Exception as e:
            st.error(f"Error retrieving persistent vector store: {e}")
    # If not found or retrieval fails, create a new one.
    vector_store = upload_and_index_file(PDF_FILE_PATH)
    with open(VECTOR_STORE_ID_FILE, "w") as f:
        f.write(vector_store.id)
    return vector_store

def duckduckgo_web_search(query):
    """Performs a search using the DuckDuckGo Instant Answer API and returns result snippets."""
    params = {
        "q": query,
        "format": "json",
        "no_html": 1,
        "skip_disambig": 1
    }
    response = requests.get("https://api.duckduckgo.com/", params=params)
    data = response.json()
    snippets = []
    if data.get("AbstractText"):
        snippets.append(data["AbstractText"])
    if data.get("RelatedTopics"):
        for topic in data["RelatedTopics"]:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"])
    return "\n".join(snippets)

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
    """Generates a clarifying question using GPT-4o based on the user's travel query."""
    prompt = (
        f"You are a travel expert. The user asked:\n\n"
        f"\"{user_question}\"\n\n"
        "What is one concise clarifying question you should ask to gather more specific travel details? "
        "Return only the question."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_answer(assistant_id, conversation_history, user_question):
    """
    Generates an answer using conversation history and the current user question.
    Uses internal context from the persistent vector store.
    If the answer indicates insufficient data, a clarifying question is generated.
    Additionally, if the query contains travel-specific keywords,
    it combines doc-based results with live DuckDuckGo data.
    """
    # Define keywords that might trigger live search
    forced_search_keywords = ["hotel", "hotels", "flight", "flights", "restaurant", "restaurants"]
    
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_question})
    
    thread = client.beta.threads.create(messages=messages)
    doc_based_answer = ""
    with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
        for event in stream:
            if event.event == 'thread.message.delta':
                for delta_block in event.data.delta.content:
                    if delta_block.type == 'text':
                        doc_based_answer += delta_block.text.value
    
    # If internal context is insufficient, generate a clarifying question.
    if "answer not available in context" in doc_based_answer.lower():
        clarifying_question = generate_clarifying_question(user_question)
        return clarifying_question
    
    # Otherwise, if the question contains certain keywords, perform a live web search.
    if any(keyword in user_question.lower() for keyword in forced_search_keywords):
        web_data = duckduckgo_web_search(user_question)
        if web_data.strip():
            combined_answer = f"{doc_based_answer}\n\nAdditional info from DuckDuckGo:\n{web_data}"
            return combined_answer
    return doc_based_answer

def main():
    apply_custom_css()  # Apply the custom CSS
    
    st.title("TravClan Navigator 🌍🧭 - Your Travel Assistant")
    st.write("Welcome! Ask about your trip, itinerary planning, or internal TravClan processes.")
    
    # Maintain conversation history in session state.
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Use persistent vector store
    if "vector_store" not in st.session_state:
        with st.spinner("Indexing travel documents..."):
            vector_store = get_persistent_vector_store()
            st.session_state.vector_store = vector_store
            assistant = create_assistant_with_vector_store(vector_store)
            st.session_state.assistant = assistant
    else:
        assistant = st.session_state.assistant
    
    # Display conversation using Streamlit's chat UI.
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
