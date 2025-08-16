# InsightAgent: Agentic PDF Chatbot

A powerful AI-powered chatbot that can analyze PDF documents and answer questions about their content. Built with Streamlit, LangChain, and LangGraph.

## Features

- **PDF Document Processing**: Upload and process multiple PDF files
- **Intelligent Q&A**: Ask specific questions about document content
- **Document Summarization**: Get comprehensive summaries of uploaded PDFs
- **Multi-Chat Support**: Create and manage multiple chat threads
- **Persistent Memory**: Chat history is maintained across sessions
- **Real-time Processing**: Fast document analysis with vector embeddings

## Quick Start

### Prerequisites

- Python 3.8+
- GROQ API Key (for LLM access)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   ```

4. **Run the application**
   ```bash
   streamlit run chatbot_app.py
   ```

The app will open in your browser at `http://localhost:8501`

## How to Use

### 1. Upload Documents
- Click on the sidebar "Upload Documents" section
- Select one or more PDF files
- Click "Process Documents" to analyze them

### 2. Start Chatting
- Once documents are processed, you can start asking questions
- Type your question in the chat input at the bottom

### 3. Example Queries
- **Questions**: "What are the main skills mentioned?", "What companies are discussed?"
- **Summaries**: "Summarize the document", "Give me an overview", "What are the key points?"

### 4. Multiple Chats
- Click "New Chat" to start a fresh conversation
- Switch between chat threads using the Chat History sidebar
- Delete unwanted threads with the 🗑️ button

## Project Structure

```
agentic-rag-chatbot/
│
├── 📄 README.md                    # Project documentation
├── 📄 requirements.txt             # Python dependencies
├── 📄 .env                        # Environment variables           
│
├── 📁 app/                        # Main application 
│   └── 📄 chat_app.py             # Main Streamlit app (your main.py)
│
├── 📁 core/                       # Core business logic
│   ├── 📄 agent_builder.py        # Agent compilation logic
│   ├── 📄 agent_state.py          # Agent state definitions
│   └── 📄 tools.py                # RAG and summarization tools
│
├── 📁 services/                   # Data processing services
│   ├── 📄 document_processor.py   # PDF processing
│   └── 📄 vector_store.py         # Vector store operations
│
├── 📁 config/                     # Configuration
    └── 📄 config.py               # App configuration

```

## Technical Details

### Core Components

- **LLM**: Uses GROQ's Llama model for natural language processing
- **Vector Store**: FAISS for efficient similarity search
- **Embeddings**: SentenceTransformers for document embeddings
- **Agent Framework**: LangGraph for complex agent workflows
- **Frontend**: Streamlit for interactive web interface

### Tools Available

1. **AnswerQuestionAboutPDFs**: Retrieval-Augmented Generation for specific questions
2. **SummarizePDF**: Document summarization using map-reduce approach

### Memory Management

The application uses LangGraph's MemorySaver to maintain:
- Chat history across sessions
- Multiple conversation threads
- Agent state persistence

## Dependencies

Key packages used in this project:

- `streamlit` - Web interface
- `langchain` - LLM framework
- `langchain-groq` - GROQ integration
- `langgraph` - Agent workflows
- `faiss-cpu` - Vector database
- `sentence-transformers` - Text embeddings
- `pypdf` - PDF processing

