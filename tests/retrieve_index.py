import os
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone

# Set the Pinecone API key
api_key = os.getenv("PINECONE_API_KEY")
print("api_key is", api_key)
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env file")

print("API Key found:", bool(api_key))

pc = Pinecone(api_key)
index = pc.Index("n8n")
print(index)

# Retrieve and print all active indexes
active_indexes = pc.list_indexes()
print("Active Pinecone Indexes:")
for index in active_indexes:
    print(index)
