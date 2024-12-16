import os
import re
import tempfile
from datetime import datetime
from typing import Dict, List

from dotenv import load_dotenv
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from pinecone import Pinecone as PineconeClient

# Load environment variables
load_dotenv()


def init_pinecone():
    """Initialize Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
    return PineconeClient(api_key=api_key)


def delete_index(index_name: str):
    """Delete an entire index."""
    try:
        pc = init_pinecone()
        pc.delete_index(index_name)
        return True
    except Exception as e:
        raise Exception(f"Error deleting index: {str(e)}")


def get_active_indexes():
    """Get list of active Pinecone indexes."""
    pc = init_pinecone()
    return [index.name for index in pc.list_indexes()]


def process_document(uploaded_file, metadata: Dict, namespace: str = ""):
    """Process uploaded document based on file type."""
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=uploaded_file.name
        ) as tmp_file:
            # Write uploaded file content to temp file
            tmp_file.write(uploaded_file.getvalue())
            temp_path = tmp_file.name

        # Select appropriate loader based on file type
        file_extension = uploaded_file.name.split(".")[-1].lower()

        if file_extension == "pdf":
            loader = PyPDFLoader(temp_path)
        elif file_extension == "md":
            loader = UnstructuredMarkdownLoader(temp_path)
            # Enhance preprocessing for markdown
            documents = loader.load()
            for doc in documents:
                # Example: Remove markdown headers
                doc.page_content = re.sub(
                    r"^#+\s", "", doc.page_content, flags=re.MULTILINE
                )
                # Example: Normalize whitespace
                doc.page_content = re.sub(r"\s+", " ", doc.page_content).strip()
        elif file_extension == "txt":
            loader = TextLoader(temp_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Load documents
        documents = loader.load()

        # Add metadata to documents
        for doc in documents:
            doc.metadata.update(metadata)
            doc.metadata.update(
                {
                    "timestamp": datetime.now().isoformat(),
                    "file_type": file_extension,
                    "source": uploaded_file.name,
                }
            )

        # Split documents into chunks with overlap
        # chunk_size: Number of characters in each chunk
        # chunk_overlap: Number of characters that overlap between chunks
        # Example: With chunk_size=1000 and chunk_overlap=200:
        #   Chunk 1: characters 0-1000
        #   Chunk 2: characters 800-1800 (overlaps with end of Chunk 1)
        #   Chunk 3: characters 1600-2600 (overlaps with end of Chunk 2)
        # This overlap ensures that concepts/phrases that might be split between chunks
        # are fully captured in at least one chunk, improving search quality
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)

        # Cleanup temporary file
        os.remove(temp_path)
        return chunks

    except Exception as e:
        if "temp_path" in locals() and os.path.exists(temp_path):
            os.remove(temp_path)
        raise Exception(f"Error processing document: {str(e)}")


def upload_to_pinecone(chunks: List, index_name: str, namespace: str = ""):
    """Upload document chunks to Pinecone."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        return Pinecone.from_documents(
            chunks, embeddings, index_name=index_name, namespace=namespace
        )
    except Exception as e:
        raise Exception(f"Error uploading to Pinecone: {str(e)}")


def delete_namespace(index_name: str, namespace: str):
    """Delete all vectors in a namespace."""
    try:
        pc = init_pinecone()
        index = pc.Index(index_name)
        index.delete(delete_all=True, namespace=namespace)
        return True
    except Exception as e:
        raise Exception(f"Error deleting namespace: {str(e)}")


def get_index_stats(index_name: str):
    """Get statistics for a Pinecone index."""
    try:
        pc = init_pinecone()
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        return stats
    except Exception as e:
        raise Exception(f"Error getting index stats: {str(e)}")


def query_index(index_name: str, query: str, namespace: str = "", top_k: int = 5):
    """Query Pinecone index."""
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
        vectorstore = Pinecone.from_existing_index(
            index_name=index_name, embedding=embeddings, namespace=namespace
        )

        docs = vectorstore.similarity_search(query, k=top_k, namespace=namespace)

        return {
            "query": query,
            "matches": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": getattr(doc, "score", None),
                }
                for doc in docs
            ],
            "namespace": namespace,
            "total_results": len(docs),
        }
    except Exception as e:
        raise Exception(f"Error querying index: {str(e)}")
