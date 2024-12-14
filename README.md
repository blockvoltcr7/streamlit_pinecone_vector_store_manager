# Streamlit Pinecone Vector Store Manager

A Streamlit application for managing and querying vector stores in Pinecone using LangChain integration.

## Getting Started

### Prerequisites
- Python 3.8+
- OpenAI API key
- Pinecone API key
- Create a index in Pinecone, default namespace is "n8n" for this project. if you want to use a different namespace, you can change it in the code.
- Git

### Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/streamlit_pinecone_vector_store_manager.git
cd streamlit_pinecone_vector_store_manager
```

2. Create and activate virtual environment
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages
- cd into root project directory
```bash
pip install -r requirements.txt
```

4. Environment Setup
- Create a `.env` file in the root directory
- Add your API keys:
```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

### Running the Application

1. Navigate to the Streamlit directory
```bash
cd streamlit
```

2. Launch the Streamlit app
```bash
streamlit run home.py
```

## Additional Recommendations

1. **API Key Security**
   - Never commit your `.env` file to version control
   - Keep your API keys secure and rotate them regularly

2. **Virtual Environment**
   - Always use a virtual environment to avoid package conflicts
   - Update requirements.txt when adding new dependencies:
     ```bash
     pip freeze > requirements.txt
     ```

3. **Memory Management**
   - Monitor your Pinecone vector store usage
   - Clean up unused vectors to optimize storage

4. **Error Handling**
   - Check API rate limits
   - Implement proper error handling for API calls

5. **Performance**
   - Use batch operations when possible
   - Implement caching for frequently accessed data

## Features
- PDF, Markdown, Text document processing
- Vector store management
- Metadata filtering
- Document querying with LangChain
- Interactive Streamlit interface

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## License
This project is licensed under the MIT License - see the LICENSE file for details
```

This README provides clear instructions for setting up and running the project, along with additional recommendations for better usage and maintenance. Feel free to modify it according to your specific needs or add more sections as required.
