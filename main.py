import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests

# Ensure you have Streamlit >= 1.74 for st.chat_xxx APIs
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
    """Performs a search using DuckDuckGo Instant Answer API.
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
    """Generates an answer using conversation history and the current user question.
       If the response indicates insufficient internal data, perform a DuckDuckGo web search.
    """
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_question})
    
    # Create a thread for the conversation.
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
    
    # If internal context is insufficient, do a live DuckDuckGo web search.
    if "answer not available in context" in answer.lower():
        web_data = duckduckgo_web_search(user_question)
        answer += "\n\nAdditional live information from DuckDuckGo:\n" + web_data

    return answer

def main():
    st.title("TravClan Navigator 🌍🧭")
    st.write("Welcome to your Travel Assistant with a Chat UI!")

    # Maintain conversation history in session state.
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []

    # Index the PDF and create the assistant only once.
    if "vector_store" not in st.session_state:
        with st.spinner("Indexing travel documents..."):
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            st.session_state.vector_store = vector_store
            assistant = create_assistant_with_vector_store(vector_store)
            st.session_state.assistant = assistant
    else:
        assistant = st.session_state.assistant

    # Display existing chat messages in session
    for msg in st.session_state.conversation_history:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])

    # Use st.chat_input for user input (Streamlit 1.74+)
    user_question = st.chat_input("Type your question here...")

    if user_question:
        with st.chat_message("user"):
            st.write(user_question)
        with st.spinner("Processing your query..."):
            answer = generate_answer(assistant.id, st.session_state.conversation_history, user_question)
        
        # Update session state with new messages
        st.session_state.conversation_history.append({"role": "user", "content": user_question})
        st.session_state.conversation_history.append({"role": "assistant", "content": answer})

        # Display the assistant's response
        with st.chat_message("assistant"):
            st.write(answer)

if __name__ == "__main__":
    main()
