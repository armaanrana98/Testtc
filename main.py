import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests
import os
from bs4 import BeautifulSoup  # Added for parsing Kayak HTML

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
    Apply custom CSS for a balanced, modern theme.
    This version uses a light container background with pastel chat bubbles,
    ensuring that text remains dark and easily readable.
    """
    st.markdown(
        """
        <style>
        /* Import Roboto font for a modern look */
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

        /* Overall app styling */
        .stApp {
            background: linear-gradient(180deg, #f9f9f9, #eaeaea);
            color: #333333;
            font-family: 'Roboto', sans-serif;
        }
        
        /* Main container styling */
        .main .block-container {
            max-width: 900px;
            background-color: #ffffff;
            border-radius: 12px;
            padding: 2rem;
            margin: 2rem auto;
            box-shadow: 0 0 15px rgba(0, 0, 0, 0.1);
        }
        
        /* Chat bubble for user messages: pastel blue background with dark text */
        .stChatMessage-user {
            background-color: #d0e6ff !important;
            color: #0d1b2a !important;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        }
        
        /* Chat bubble for assistant messages: pastel green background with dark text */
        .stChatMessage-assistant {
            background-color: #dfffd8 !important;
            color: #0d1b2a !important;
            border-radius: 10px;
            padding: 1rem;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
            box-shadow: 0 0 8px rgba(0, 0, 0, 0.05);
        }
        
        /* Chat input area styling */
        .stChatInput {
            background-color: #ffffff !important;
            border-top: 1px solid #cccccc;
            padding: 1rem;
        }
        
        /* Chat input text box styling: light gray background with dark text */
        .stChatInput textarea {
            background-color: #f4f4f4 !important;
            color: #333333 !important;
            border: 1px solid #cccccc !important;
            border-radius: 8px !important;
            font-family: 'Roboto', sans-serif;
            padding: 0.6rem !important;
            font-size: 1rem;
        }
        
        /* Headings accent color */
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
    """Retrieves the persistent vector store using the provided vector store ID."""
    try:
        vector_store = client.vector_stores.retrieve(PERSISTENT_VECTOR_STORE_ID)
        return vector_store
    except Exception as e:
        st.error(f"Error retrieving persistent vector store: {e}")
        return None

def kayak_hotel_search(location, checkin_date, checkout_date):
    """
    Searches for hotels on Kayak for a given location and date range.
    
    Parameters:
    - location: A string indicating the destination (e.g., "new-york").
    - checkin_date: Check-in date in the format YYYY-MM-DD.
    - checkout_date: Check-out date in the format YYYY-MM-DD.
    
    Returns:
    - A list of dictionaries containing hotel information.
    """
    # Construct the search URL (adjust the URL pattern as needed)
    search_url = f"https://www.kayak.com/hotels/{location}/{checkin_date}/{checkout_date}"
    
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/91.0.4472.124 Safari/537.36")
    }
    
    response = requests.get(search_url, headers=headers)
    hotels = []
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Note: The CSS selectors below are indicative.
        # You may need to adjust these based on Kayak's current HTML structure.
        for hotel_card in soup.find_all("div", class_="hotelCard"):
            try:
                name = hotel_card.find("span", class_="hotelName").get_text(strip=True)
                price = hotel_card.find("span", class_="price").get_text(strip=True)
                hotels.append({"name": name, "price": price})
            except AttributeError:
                continue
    else:
        st.error("Error: Unable to fetch hotel data from Kayak.")
    
    return hotels

def kayak_flight_search(origin, destination, depart_date, return_date):
    """
    Searches for flights on Kayak between two destinations.
    
    Parameters:
    - origin: Departure airport or city code.
    - destination: Arrival airport or city code.
    - depart_date: Departure date in YYYY-MM-DD format.
    - return_date: Return date in YYYY-MM-DD format.
    
    Returns:
    - A list of dictionaries containing flight details.
    """
    # Construct the flight search URL (adjust the URL pattern as needed)
    search_url = f"https://www.kayak.com/flights/{origin}-{destination}/{depart_date}/{return_date}"
    
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/91.0.4472.124 Safari/537.36")
    }
    
    response = requests.get(search_url, headers=headers)
    flights = []
    
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        # Note: These CSS selectors are placeholders and might need updating.
        for flight_card in soup.find_all("div", class_="flightCard"):
            try:
                airline = flight_card.find("span", class_="airline").get_text(strip=True)
                cost = flight_card.find("span", class_="price").get_text(strip=True)
                flights.append({"airline": airline, "price": cost})
            except AttributeError:
                continue
    else:
        st.error("Error: Unable to fetch flight data from Kayak.")
    
    return flights

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
    3. Additionally, if the user query contains specific travel keywords ("hotel" or "flight"), 
       perform live Kayak searches and append those results.
    """
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

    # Fallback to a generic itinerary if internal content is insufficient.
    if "answer not available in context" in doc_based_answer.lower():
        doc_based_answer = generate_generic_itinerary(user_question)

    additional_results = ""
    
    # Incorporate Kayak hotel search if the query contains "hotel"
    if "hotel" in user_question.lower():
        # Hardcoded example parameters; customize or extract from user input as needed.
        hotels = kayak_hotel_search(location="new-york", checkin_date="2025-06-01", checkout_date="2025-06-05")
        if hotels:
            additional_results += "\n\nTop Hotels:\n"
            for hotel in hotels[:5]:  # Limit to top 5 results
                additional_results += f"- {hotel['name']}: {hotel['price']}\n"
    
    # Incorporate Kayak flight search if the query contains "flight"
    if "flight" in user_question.lower():
        flights = kayak_flight_search(origin="JFK", destination="LAX", depart_date="2025-06-01", return_date="2025-06-05")
        if flights:
            additional_results += "\n\nTop Flights:\n"
            for flight in flights[:5]:
                additional_results += f"- {flight['airline']}: {flight['price']}\n"

    if additional_results:
        return f"{doc_based_answer}\n\n{additional_results}"
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
