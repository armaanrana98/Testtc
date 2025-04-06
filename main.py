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
# If using persistent storage, you could store vector store ID externally.
# For demonstration, we'll assume the persistent vector store is retrieved via our function.

# Retrieve API key from Streamlit secrets.
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client with beta headers.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def apply_custom_css():
    """
    Apply custom CSS for a balanced, modern theme with high readability.
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
    """Uploads and indexes the PDF document into an OpenAI vector store.
    See: https://platform.openai.com/docs/api-reference/vector-stores
    """
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
    """
    Creates an assistant that uses the persistent vector store for context.
    Instructions:
      - You are TravClan Navigator Assistant, a highly knowledgeable travel assistant.
      - Use the provided internal travel documents to answer questions about itinerary planning,
        booking processes, and internal TravClan protocols.
      - If the internal documents do not contain sufficient information to fully answer the query,
        reply with "answer not available in context".
    """
    assistant = client.beta.assistants.create(
        name="TravClan Navigator Assistant",
        instructions=(
            "You are TravClan Navigator Assistant, a highly knowledgeable travel expert representing TravClan. "
            "Use the provided internal travel documents to answer questions about itinerary planning, "
            "booking processes, and internal TravClan protocols with accuracy and detail. "
            "If the internal documents do not contain sufficient information to answer the query fully, "
            "respond with 'answer not available in context'."
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
    Uses GPT-4o to generate a single, concise clarifying question that helps gather
    the additional travel details needed for a complete answer.
    """
    prompt = (
        f"You are a seasoned travel expert. The user asked:\n\n"
        f"\"{user_question}\"\n\n"
        "Provide one clear and concise clarifying question to obtain more specific details (e.g., duration, dates, number of nights) that will help you generate a complete and accurate itinerary or travel recommendation. "
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
    Uses GPT-4o to generate a best-effort itinerary or travel recommendation when the internal
    documents do not provide sufficient details.
    """
    prompt = (
        f"You are an expert travel planner with extensive experience in creating detailed itineraries. "
        f"The user has asked: \"{user_question}\". "
        "The internal documents do not contain enough specific details for this query. "
        "Please generate a detailed and helpful travel itinerary or recommendation based on your general travel knowledge, "
        "including key attractions, suggested durations, and travel tips. "
        "Return the itinerary in a structured, easy-to-read format."
    )
    response = client.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500
    )
    return response["choices"][0]["message"]["content"].strip()

def generate_answer(assistant_id, conversation_history, user_question):
    """
    Generates an answer using conversation history and the current user question.
    Steps:
      1. Retrieve an answer using internal document context.
      2. If the answer contains "answer not available in context", then generate a generic itinerary.
      3. Additionally, if the query includes travel-specific keywords (e.g., hotels, flights),
         perform a live DuckDuckGo web search and append the results.
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

    # If internal documents don't have sufficient details, generate a generic itinerary.
    if "answer not available in context" in doc_based_answer.lower():
        generic_response = generate_generic_itinerary(user_question)
        return generic_response

    # If the query contains travel-specific keywords, perform a live web search and combine results.
    if any(keyword in user_question.lower() for keyword in forced_search_keywords):
        web_data = duckduckgo_web_search(user_question)
        if web_data.strip():
            combined_answer = f"{doc_based_answer}\n\nAdditional information from DuckDuckGo:\n{web_data}"
            return combined_answer

    return doc_based_answer

def main():
    apply_custom_css()  # Apply custom styling

    st.title("TravClan Navigator 🌍🧭 - Your Travel Assistant")
    st.write("Welcome! Ask about your trip, itinerary planning, or internal TravClan processes.")
    
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    
    # Retrieve persistent vector store (assumed to be persistent in production).
    if "vector_store" not in st.session_state:
        with st.spinner("Retrieving travel documents..."):
            # For production, you would retrieve a persistent vector store by its ID.
            # Here we create a new one if not found.
            vector_store = upload_and_index_file(PDF_FILE_PATH)
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
