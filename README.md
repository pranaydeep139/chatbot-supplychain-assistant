# Supply Chain AI Agent

A modern, full-stack chatbot application for supply chain analytics and document Q&A, built with FastAPI, DuckDB, LangChain, Qdrant, and a beautiful custom UI. The agent can answer questions using both a large supply chain dataset and a library of supply chain policy documents, providing actionable business insights.

---

## Features

- **User Authentication**: Secure registration and login system.
- **Real-Time Chat**: Modern, responsive chat UI with user/AI avatars.
- **Hybrid Q&A**: Answers questions using both SQL (DuckDB) and document retrieval (Qdrant + LangChain).
- **Business Analysis**: AI provides actionable, business-focused recommendations.
- **Document Search**: Upload and index supply chain policy PDFs for semantic search.
- **Data Analytics**: Query a large supply chain CSV with natural language.
- **Extensible**: Easily add new documents, data, or response logic.

---

## Architecture Overview

- **Backend**: FastAPI
  - User/session management (DuckDB)
  - Chat endpoint: routes questions to SQL, document retrieval, or both
  - Integrates with Google Gemini API for LLM responses
  - Uses LangChain for embeddings and Qdrant for vector search
- **Frontend**: Jinja2 templates + custom HTML/CSS/JS
  - Responsive, modern chat interface
  - Login and registration pages
- **Data**:
  - `DataCoSupplyChainDataset_UTF8.csv`: Main supply chain dataset (loaded into DuckDB)
  - `supply_docs/`: Folder of supply chain policy PDFs (indexed for semantic search)

---

## Setup & Installation

1. **Clone the repository**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   - Create a `.env` file in the project root:
     ```env
     GEMINI_API_KEY=your_google_gemini_api_key
     QDRANT_URL=https://your-qdrant-instance.com
     QDRANT_API_KEY=your_qdrant_api_key
     SECRET_KEY=your_random_secret
     ```
   - Or set them in your shell.

4. **Prepare the data**
   - Place your main CSV as `DataCoSupplyChainDataset_UTF8.csv` in the project root.
   - Place your policy PDFs in the `supply_docs/` folder.

5. **Index the documents**
   - Run the document loader to embed and index PDFs:
     ```bash
     python doc_loader.py
     ```

6. **Run the application**
   ```bash
   uvicorn main:app --reload
   ```
   - Open your browser at [http://localhost:8000](http://localhost:8000)

---

## Usage

- Register a new account or log in.
- Type your question in the chat (e.g., "What are the top supply chain risks last year?" or "Show me the policy for supplier selection.")
- The agent will answer using data, documents, or both, and provide business recommendations.
- All chat history is stored per session.

---

## Extending the Chatbot

- **Add new documents**: Place new PDFs in `supply_docs/` and re-run `doc_loader.py`.
- **Change data**: Replace the CSV and restart the app.
- **Customize prompts/logic**: Edit the main logic in `main.py` (see the `/chat` endpoint for prompt structure and process flow).

---

## File/Folder Structure

```
├── main.py                # FastAPI backend, chat logic, authentication
├── doc_loader.py          # Script to index PDFs into Qdrant
├── del_docs.py            # Script to delete Qdrant collection
├── requirements.txt       # Python dependencies
├── DataCoSupplyChainDataset_UTF8.csv  # Main supply chain data
├── supply_docs/           # Folder of policy PDFs
├── templates/
│   ├── chat.html          # Chat UI
│   ├── login.html         # Login page
│   └── register.html      # Registration page
├── static/                # Static assets (CSS, JS)
├── .env                   # (Not committed) Environment variables
```

---

## Technologies Used

- FastAPI
- DuckDB
- LangChain & LangChain-Community
- Qdrant (vector database)
- HuggingFace Transformers (embeddings)
- Google Gemini API (LLM)
- Jinja2 Templates
- HTML/CSS/JavaScript

---

## License

MIT License (or specify your own)

---

## Acknowledgements

- [LangChain](https://langchain.com/)
- [Qdrant](https://qdrant.tech/)
- [Google Gemini](https://ai.google.dev/)
- [FastAPI](https://fastapi.tiangolo.com/) 