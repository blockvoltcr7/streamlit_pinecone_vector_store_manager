import os
from pathlib import Path
from time import sleep

from dotenv import load_dotenv
from pinecone import Pinecone

# Note: The free Pinecone plan does not include the ability to create an index.


# Load environment variables
load_dotenv()


def init_pinecone():
    """Initialize Pinecone client"""
    # Get Pinecone API key
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")

    return Pinecone(api_key=api_key)


def create_index(index_name: str = "pro-roofers"):
    """Create a new Pinecone index"""
    try:
        pc = init_pinecone()

        # Check if index already exists
        if index_name in pc.list_indexes().names():
            print(f"Index '{index_name}' already exists")
            return

        # Create index with specified configuration
        index = pc.create_index(
            name=index_name,
            dimension=1536,  # OpenAI embedding dimension
            metric="cosine",
            spec={"serverless": {"cloud": "aws", "region": "us-west-2"}},
        )

        print(f"Index '{index_name}' created successfully")
        print("\nWaiting for index to be ready...")

        # Wait for index to be ready
        while not index_name in pc.list_indexes().names():
            sleep(1)

        print(f"Index '{index_name}' is ready")

        # Get and print index details
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print("\nIndex stats:")
        print(stats)

    except Exception as e:
        print(f"Error creating index: {str(e)}")


def main():
    # Create index
    create_index("test-index")

    # List all indexes
    pc = init_pinecone()
    print("\nAll available indexes:")
    indexes = pc.list_indexes()
    for index in indexes:
        print(index)


if __name__ == "__main__":
    main()
