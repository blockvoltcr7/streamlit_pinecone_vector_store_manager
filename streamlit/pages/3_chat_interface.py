import os
from typing import Dict

from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Pinecone
from utils.pinecone_utils import get_active_indexes, get_index_stats

import streamlit as st

# Load environment variables
load_dotenv()

# Initialize memory for conversation
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def init_session_state():
    """Initialize chat session state."""
    if "conversation" not in st.session_state:
        st.session_state.conversation = []


def query_vector_store(vector_store, query: str):
    """Query vector store and get conversational response using LLM."""
    try:
        # Initialize OpenAI chat model
        llm = ChatOpenAI(
            temperature=0.7,  # More conversational creativity
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4o",
        )

        # Setup conversational chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(),
            memory=memory,  # Attach memory for conversation
            verbose=True,
        )
        # Get response from the chain
        response = chain.run({"question": query})

        return response

    except Exception as e:
        st.error(f"Error querying vector store: {str(e)}")
        return None


def render_chat_interface(vector_store):
    st.subheader("Chat Interface")

    # Display conversation history
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  # Use markdown for better formatting

    # Chat input
    if prompt := st.chat_input(
        "Ask a question or start a conversation about your documents..."
    ):
        # Add user message to conversation
        st.session_state.conversation.append({"role": "user", "content": prompt})

        with st.spinner("Generating response..."):
            # Get conversational response
            response = query_vector_store(vector_store, prompt)

            if response:
                # Add assistant response to conversation
                st.session_state.conversation.append(
                    {"role": "assistant", "content": response}
                )

                # Force refresh to show new messages
                st.rerun()


def chat_interface_page():
    st.title("Chat with Documents")
    init_session_state()

    try:
        indexes = get_active_indexes()
        if not indexes:
            st.error("No Pinecone indexes available")
            return

        # Add settings to sidebar
        with st.sidebar:
            st.header("Settings")
            selected_index = st.selectbox("Select Index", indexes)

            # Get available namespaces for the selected index
            try:
                stats = get_index_stats(selected_index)
                available_namespaces = list(stats.namespaces.keys())

                # Replace empty namespace with "default" for display
                available_namespaces = [
                    "default" if ns == "" else ns for ns in available_namespaces
                ]

                if not available_namespaces:
                    st.warning("No namespaces found in this index")
                    return

                # Namespace selection dropdown
                namespace = st.selectbox(
                    "Select Namespace",
                    options=available_namespaces,
                    help="Select a namespace to chat with",
                )

                # Convert "default" back to empty string for Pinecone
                namespace = "" if namespace == "default" else namespace

            except Exception as e:
                st.error(f"Error fetching namespaces: {str(e)}")
                return

        # Initialize vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Pinecone.from_existing_index(
            index_name=selected_index,
            embedding=embeddings,
            namespace=namespace,
        )

        # Render chat interface
        render_chat_interface(vector_store)

    except Exception as e:
        st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    chat_interface_page()
