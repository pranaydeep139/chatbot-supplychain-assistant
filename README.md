# Simple FastAPI Chatbot

A simple chatbot application built with FastAPI and a clean, modern UI.

## Features

- Real-time chat interface
- Simple response system
- Modern and responsive UI
- Easy to extend and customize

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key as an environment variable (recommended) or in a `.env` file:
```bash
# .env file (create in project root)
OPENAI_API_KEY=sk-YOUR_OPENAI_KEY
```
Or set it in your shell:
```bash
set OPENAI_API_KEY=sk-YOUR_OPENAI_KEY  # Windows
export OPENAI_API_KEY=sk-YOUR_OPENAI_KEY  # Mac/Linux
```

3. Run the application:
```bash
uvicorn main:app --reload
```

4. Open your browser and navigate to `http://localhost:8000`

## Usage

- Type your message in the input field
- Press Enter or click the Send button to send your message
- The chatbot will respond based on predefined patterns
- Current supported keywords: "hello", "how are you", "bye"

## Extending the Chatbot

To add more responses, modify the `responses` dictionary in `main.py`. Add new key-value pairs where:
- Key: The trigger word or phrase
- Value: The bot's response

## Technologies Used

- FastAPI
- Jinja2 Templates
- HTML/CSS
- JavaScript 