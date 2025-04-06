import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests

st.set_page_config(
    page_title="TravClan Navigator 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"

# Retrieve API key from Streamlit secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client with beta headers for assistants and vector stores.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
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
    """Uploads and indexes the PDF document into an OpenAI vector store.
    See: https://platform.openai.com/docs/api-reference/vector-stores
    """
    with open(pdf_file_path, "rb") as file_stream:
        # Create a vector store for your travel documents.
        vector_store = client.vector_stores.create(name="TravClan Navigator Documents")
        # Upload and index the file.
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file_stream]
        )
    return vector_store

def duckduckgo_web_search(query):
    """Performs a search using the DuckDuckGo Instant Answer API.
       Returns top result snippets as a combined string.
    """
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

def generate_answer(assistant_id, conversation_history, user_question):
    """
    Generates an answer using conversation history and the current user question.
    If the response indicates insufficient internal data (by including "answer not available in context"),
    it asks a clarifying follow-up question, e.g., "How many nights are you planning to stay?"
    """
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_question})
    
    # Create a conversation thread.
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
    
    # Check if internal documents lack the required details.
    if "answer not available in context" in answer.lower():
        # Instead of performing a web search, ask a clarifying question.
        clarifying_question = ("It appears that our internal documents don't contain specific travel details for your query. "
                                "Could you please tell me how many nights you plan to stay?")
        answer = clarifying_question

    return answer

def main():
    st.title("TravClan Navigator 🌍🧭 - Your Travel Assistant")
    
    # Use session state to store conversation history.
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Upload and index the PDF and create the assistant only once.
    if "vector_store" not in st.session_state:
        with st.spinner("Indexing travel documents..."):
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            st.session_state.vector_store = vector_store
            assistant = create_assistant_with_vector_store(vector_store)
            st.session_state.assistant = assistant
    else:
        assistant = st.session_state.assistant
    
    # Display existing chat messages using Streamlit's chat UI.
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
    
    # Use st.chat_input for user input (requires Streamlit >= 1.74)
    user_question = st.chat_input("Type your travel question here...")
    
    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.spinner("Processing your query..."):
            answer = generate_answer(assistant.id, st.session_state.conversation_history, user_question)
        # Update conversation history.
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()
