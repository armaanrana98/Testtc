import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests
import os

st.set_page_config(
    page_title="Travvy 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"
PERSISTENT_VECTOR_STORE_ID = "vs_67f2be79b01c819198749cdd69887e11" 

# Retrieve API key from Streamlit secrets.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client with beta headers.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def apply_custom_css():
    """
    Ensures chat response text is clearly visible on a black/dark background
    with accent colors for a clean, premium look.
    """
    st.markdown(
        """
        <style>
        /* Import a modern font */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        /* Overall app styling (black bg, white text) */
        .stApp {
            background-color: #000000;
            color: #ffffff;
            font-family: 'Roboto', sans-serif;
        }

        /* Main container with dark gray background and subtle shadow */
        .main .block-container {
            max-width: 900px;
            background-color: #1a1a1a;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.7);
        }

        /* Force white text within chat messages (Markdown can override color otherwise) */
        .stChatMessage .stMarkdown, 
        .stChatMessage .stMarkdown p, 
        .stChatMessage .stMarkdown span, 
        .stChatMessage .stMarkdown div {
            color: #ffffff !important;
        }

        /* Make sure links in messages are visible */
        .stChatMessage .stMarkdown a {
            color: #ffb000 !important; /* Accent color for links */
            text-decoration: none;
        }
        .stChatMessage .stMarkdown a:hover {
            text-decoration: underline;
        }

        /* User messages: slightly lighter dark background + left accent border */
        .stChatMessage-user {
            background-color: #2a2a2a !important;
            border-left: 4px solid #ffb000 !important; 
            border-radius: 8px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        /* Assistant messages: a different shade + accent border */
        .stChatMessage-assistant {
            background-color: #333333 !important;
            border-left: 4px solid #1f78ff !important; /* Blue accent for the assistant */
            border-radius: 8px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
        }

        /* Chat input area: black background, subtle top border */
        .stChatInput {
            background-color: #000000 !important;
            border-top: 1px solid #444444;
            padding: 1rem;
        }

        /* Chat input textarea: dark gray with white text */
        .stChatInput textarea {
            background-color: #1a1a1a !important;
            color: #ffffff !important;
            border: 1px solid #444444 !important;
            border-radius: 8px !important;
            font-family: 'Roboto', sans-serif;
            padding: 0.6rem !important;
            font-size: 1rem;
            width: 100% !important;
        }

        /* Placeholder text in the chat input */
        .stChatInput textarea::placeholder {
            color: #888888; 
        }

        /* Send button: golden accent */
        .stChatInput button {
            background-color: #ffb000 !important;
            color: #000000 !important;
            border: none !important;
            border-radius: 8px !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 0.6rem 1rem !important;
            margin-left: 0.5rem !important;
            cursor: pointer;
        }
        .stChatInput button:hover {
            background-color: #e0a000 !important;
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
    """Retrieves the persistent vector store using the provided vector store ID."""
    try:
        vector_store = client.vector_stores.retrieve(PERSISTENT_VECTOR_STORE_ID)
        return vector_store
    except Exception as e:
        st.error(f"Error retrieving persistent vector store: {e}")
        return None

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
    """
    Creates an assistant that uses the persistent vector store for context.
    The assistant is given production-grade instructions.
    """
    assistant = client.beta.assistants.create(
        name="TravClan Navigator Assistant",
        instructions=(
            "You are TravClan Navigator Assistant, a highly knowledgeable travel expert representing TravClan. "
            "Use the provided internal travel documents to answer queries regarding itinerary planning, booking processes, "
            "and internal TravClan protocols with precision and detail. "
            "If the internal documents do not contain sufficient details, reply with 'answer not available in context'."
        ),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={
            "file_search": {"vector_store_ids": [vector_store.id]}
        }
    )
    return assistant

def generate_clarifying_question(user_question):
    """
    Uses GPT-4o to generate a single, concise clarifying question that gathers additional details
    necessary to produce a complete travel recommendation.
    """
    prompt = (
        f"You are a seasoned travel expert. The user asked:\n\n"
        f"\"{user_question}\"\n\n"
        "What is one clear and concise clarifying question you should ask to gather more specific travel details (e.g., dates, duration, number of nights) that will help you generate a complete itinerary or travel recommendation? "
        "Return only the question."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_generic_itinerary(user_question):
    """
    Uses GPT-4o to generate a best-effort itinerary when internal documents lack sufficient details.
    """
    prompt = (
        f"You are an expert travel planner. The user asked: \"{user_question}\". "
        "The internal documents do not have enough specific details for this query. "
        "Please create a detailed, step-by-step travel itinerary that includes key attractions, recommended durations, and travel tips, presented in a clear format."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_answer(assistant_id, conversation_history, user_question):
    """
    Generates an answer using conversation history and the user's current question.
    1. Retrieves an answer based on internal document context.
    2. If the internal content is insufficient (i.e. returns 'answer not available in context'),
       then generate a generic itinerary using GPT-4o.
    3. Additionally, if the user query contains specific travel keywords,
       perform a live DuckDuckGo search and append those results.
    """
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

    if "answer not available in context" in doc_based_answer.lower():
        generic_response = generate_generic_itinerary(user_question)
        return generic_response

    if any(keyword in user_question.lower() for keyword in forced_search_keywords):
        web_data = duckduckgo_web_search(user_question)
        if web_data.strip():
            combined_answer = f"{doc_based_answer}\n\nAdditional information from DuckDuckGo:\n{web_data}"
            return combined_answer

    return doc_based_answer

def main():
    apply_custom_css()  # Apply custom styling

    st.title("Travvy 🌍🧭 - Your Travel Assistant")
    st.write("Welcome! Ask about your trip, itinerary planning, or internal TravClan processes.")

    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Retrieve persistent vector store using provided ID.
    if "vector_store" not in st.session_state:
        with st.spinner("Retrieving travel documents..."):
            vector_store = client.vector_stores.retrieve(PERSISTENT_VECTOR_STORE_ID)
            st.session_state.vector_store = vector_store
            assistant = create_assistant_with_vector_store(vector_store)
            st.session_state.assistant = assistant
    else:
        assistant = st.session_state.assistant

    # Display existing conversation using Streamlit's chat UI.
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
