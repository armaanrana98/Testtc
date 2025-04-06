import streamlit as st
from openai import OpenAI
import PyPDF2
import mysql.connector
from mysql.connector import Error
import time

st.set_page_config(
    page_title="TravGPT",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Initialize the OpenAI client.
# We use the GA endpoints for vector stores (client.vector_stores)
# and retain beta endpoints for assistants and threads.
client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def get_db_connection():
    return mysql.connector.connect(
        host=st.secrets["mysql"]["host"],
        user=st.secrets["mysql"]["user"],
        password=st.secrets["mysql"]["password"],
        database=st.secrets["mysql"]["database"]
    )

def log_question_and_response(question, response, response_time_seconds):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        query = """
            INSERT INTO question_logs (question, response, response_time_seconds)
            VALUES (%s, %s, %s)
        """
        cursor.execute(query, (question, response, response_time_seconds))
        connection.commit()
        cursor.close()
        connection.close()
    except Error as e:
        st.error(f"Error: {e}")

def pdf_file_to_text(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def upload_and_index_file(pdf_file_path):
    with open(pdf_file_path, "rb") as file_stream:
        # Create the vector store using the GA endpoint.
        vector_store = client.vector_stores.create(name="TravGPT Documents")
        # Upload the file using the file batching method.
        file_batch = client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file_stream]
        )
    return vector_store

def create_assistant_with_vector_store(vector_store):
    assistant = client.beta.assistants.create(
        name="TravGPT Assistant",
        instructions=(
            "Answer the question as precisely as possible using the provided context. "
            "Answer it in a proper and detailed manner. If the answer is not contained in the context, "
            "say 'answer not available in context'."
        ),
        model="gpt-4o",
        tools=[{"type": "file_search"}],
        tool_resources={"file_search": {"vector_store_ids": [vector_store.id]}}
    )
    return assistant

def generate_answer(assistant_id, question):
    thread = client.beta.threads.create(
        messages=[{"role": "user", "content": question}]
    )

    answer = ""
    start_time = time.time()
    with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
        for event in stream:
            if event.event == 'thread.message.delta':
                for delta_block in event.data.delta.content:
                    if delta_block.type == 'text':
                        answer += delta_block.text.value
    end_time = time.time()
    response_time_seconds = end_time - start_time
    log_question_and_response(question, answer, response_time_seconds)
    st.write(answer)

def main():
    st.header("TravGPT🤖")

    question = st.text_input("Get answers to all your internal queries:")

    if st.button("Ask"):
        with st.spinner("Processing..."):
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            assistant = create_assistant_with_vector_store(vector_store)

            if question:
                generate_answer(assistant.id, question)
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()
