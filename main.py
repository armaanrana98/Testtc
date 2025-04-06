import streamlit as st
from openai import OpenAI
import PyPDF2
import time
import requests
import uuid

st.set_page_config(
    page_title="TravClan Navigator 🌍🧭",
    page_icon="🌍🧭",
    layout="centered",
    initial_sidebar_state="auto"
)

PDF_FILE_PATH = "data.pdf"

openai_api_key = st.secrets["OPENAI_API_KEY"]

client = OpenAI(
    api_key=openai_api_key,
    default_headers={"OpenAI-Beta": "assistants=v2"}
)

def pdf_file_to_text(pdf_file):
    text = ""
    with open(pdf_file, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def upload_and_index_file(pdf_file_path):
    with open(pdf_file_path, "rb") as file_stream:
        vector_store = client.vector_stores.create(name="TravClan Navigator Documents")
        client.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id,
            files=[file_stream]
        )
    return vector_store

def create_assistant_with_vector_store(vector_store):
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
    messages = conversation_history.copy()
    messages.append({"role": "user", "content": user_question})
    
    thread = client.beta.threads.create(messages=messages)
    answer = ""
    with client.beta.threads.runs.stream(thread_id=thread.id, assistant_id=assistant_id) as stream:
        for event in stream:
            if event.event == 'thread.message.delta':
                for delta_block in event.data.delta.content:
                    if delta_block.type == 'text':
                        answer += delta_block.text.value
    
    if "answer not available in context" in answer.lower():
        clarifying_question = generate_clarifying_question(user_question)
        answer = clarifying_question

    return answer

def apply_chat_widget_css():
    """
    This CSS + HTML layout tries to replicate a floating chat widget in the bottom-right corner
    with a pink top bar, small bubble style, etc.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap');
    body, .stApp {
        background-color: #f1f1f1;
        font-family: 'Orbitron', sans-serif;
    }
    /* Container for entire chat widget */
    #chatWidget {
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 350px;
        height: 500px;
        background-color: #fff;
        border-radius: 10px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        display: flex;
        flex-direction: column;
        z-index: 9999;
    }
    /* Pink top bar */
    #chatHeader {
        background-color: #ff5a9f;
        color: #fff;
        padding: 10px;
        border-radius: 10px 10px 0 0;
        text-align: center;
        font-weight: bold;
    }
    /* Chat body where messages appear */
    #chatBody {
        flex: 1;
        padding: 10px;
        overflow-y: auto;
        background-color: #f9f9f9;
    }
    /* Chat message bubble style */
    .chat-bubble {
        max-width: 80%;
        margin-bottom: 10px;
        padding: 8px 12px;
        border-radius: 16px;
        line-height: 1.4;
        font-size: 0.95rem;
        color: #333;
    }
    .chat-bubble.user {
        background-color: #d1e8ff;
        margin-left: auto;
    }
    .chat-bubble.assistant {
        background-color: #ebebeb;
        margin-right: auto;
    }
    /* Chat footer with input and button */
    #chatFooter {
        padding: 10px;
        border-top: 1px solid #ccc;
        background-color: #f3f3f3;
    }
    #chatInputBox {
        width: calc(100% - 60px);
        padding: 8px;
        border-radius: 4px;
        border: 1px solid #ccc;
        font-family: 'Orbitron', sans-serif;
    }
    #sendBtn {
        background-color: #ff5a9f;
        color: #fff;
        border: none;
        padding: 8px 16px;
        border-radius: 4px;
        cursor: pointer;
        margin-left: 5px;
        font-family: 'Orbitron', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.title("TravClan Navigator 🌍🧭 (Page)")  # Title for the overall page
    st.write("This is your main page content. The chat widget floats at the bottom-right corner.")

    # If we haven't created a vector store yet, do so.
    if "vector_store" not in st.session_state:
        with st.spinner("Indexing travel documents..."):
            vector_store = upload_and_index_file(PDF_FILE_PATH)
            st.session_state.vector_store = vector_store
            st.session_state.assistant = create_assistant_with_vector_store(vector_store)
    else:
        pass  # already created

    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    # Insert the floating chat widget HTML
    apply_chat_widget_css()
    st.markdown("""
    <div id="chatWidget">
      <div id="chatHeader">
        <span>Chat bot</span>
      </div>
      <div id="chatBody">
        <!-- This is where messages will appear -->
    """, unsafe_allow_html=True)

    # Display existing messages from session_state.conversation
    for message in st.session_state.conversation:
        role = message["role"]
        content = message["content"]
        bubble_class = "assistant" if role == "assistant" else "user"
        st.markdown(f"""
          <div class="chat-bubble {bubble_class}">
            {content}
          </div>
        """, unsafe_allow_html=True)

    st.markdown("""
      </div>
      <div id="chatFooter">
        <form action="#" method="post" onsubmit="var inputBox = document.getElementById('chatInputBox'); 
                                               window.parent.streamlitSendMessage(inputBox.value);
                                               inputBox.value=''; 
                                               return false;">
          <input type="text" id="chatInputBox" placeholder="Ask anything..." />
          <button id="sendBtn" type="submit">Send</button>
        </form>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Use a hidden Streamlit input to handle messages from the chat form
    # We'll store them in st.session_state and re-render
    user_msg = st.text_input("Hidden Chat Input", key="hidden_chat_input", label_visibility="collapsed")
    
    # This hack: We'll define a custom JavaScript to capture form submit from the HTML
    # and set st.session_state["hidden_chat_input"] to the user text.
    # But this approach is quite advanced and might require a custom Streamlit component.
    # Instead, we can do a simpler approach with a direct form submission.

    # We'll do a quick approach:
    # If the user typed something in hidden_chat_input, handle it.
    if user_msg:
        # Add user message
        st.session_state.conversation.append({"role": "user", "content": user_msg})
        # Generate answer
        assistant_id = st.session_state.assistant.id
        answer = generate_answer(
            assistant_id,
            st.session_state.conversation,
            user_msg
        )
        st.session_state.conversation.append({"role": "assistant", "content": answer})
        # Force rerun so we see the updated conversation
        st.experimental_rerun()

if __name__ == "__main__":
    main()
