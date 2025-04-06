import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests

# Set up page configuration.
st.set_page_config(
    page_title="TravClan Navigator 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"

# Retrieve API key from Streamlit secrets.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client with beta headers for assistants and vector stores.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def apply_custom_css():
    """
    Apply custom CSS for a futuristic dark theme with neon accents.
    """
    st.markdown(
        """
        <style>
        /* Import futuristic font */
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');

        /* Overall app background */
        .stApp {
            background: linear-gradient(135deg, #1B1E34, #23233A 70%);
            color: #EAEAEA;
            font-family: 'Orbitron', sans-serif;
        }
        
        /* Main container styling */
        .main .block-container {
            max-width: 900px;
            background: rgba(36, 37, 62, 0.85);
            border-radius: 10px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
        }

        /* Chat bubble for user messages */
        .stChatMessage-user {
            background-color: #2F2C49 !important;
            color: #FFFFFF !important;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            padding: 1rem;
            font-size: 1.05rem;
            box-shadow: 0 0 10px rgba(173, 216, 230, 0.2);
        }
        
        /* Chat bubble for assistant messages */
        .stChatMessage-assistant {
            background-color: #403B5C !important;
            color: #F5F5F5 !important;
            border-radius: 10px;
            margin-bottom: 0.5rem;
            padding: 1rem;
            font-size: 1.05rem;
            box-shadow: 0 0 10px rgba(255, 160, 122, 0.2);
        }
        
        /* Chat input area styling */
        .stChatInput {
            background-color: #1B1E34 !important;
            border-top: 1px solid #2F2C49;
        }
        
        /* Heading styling with accent */
        h1, h2, h3, h4, h5, h6 {
            color: #FDB813;
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
    """Uploads and indexes the PDF into an OpenAI vector store."""
    with open(pdf_file_path, "rb") as file_stream:
        vector_store = client.vector_stores.create(name="TravClan Navigator Documents")
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file_stream]
        )
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
    """Creates an assistant using the vector store for context."""
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
        "What is one concise clarifying question you should ask to gather more travel details? "
        "Return only the question."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=50
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_answer(assistant_id, conversation_history, user_question):
    """Generates an answer using conversation history and the current user question.
       If the response indicates insufficient internal data, a clarifying question is generated.
    """
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
    apply_custom_css()  # Apply the futuristic custom CSS
    
    st.title("TravClan Navigator 🌍🧭 - Your Travel Assistant")
    st.write("Welcome! Ask about your trip, itinerary planning, or internal TravClan processes.")
    
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
    
    # Display the conversation using Streamlit's chat UI.
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
