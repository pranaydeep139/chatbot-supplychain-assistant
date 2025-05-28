import os
import json
import uuid
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from dotenv import load_dotenv
from passlib.context import CryptContext
from langchain_community.embeddings import HuggingFaceEmbeddings
from google import genai
from langchain.vectorstores import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
import duckdb

# Load environment variables
load_dotenv()

# Initialize the Gen AI client with the API key
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Initialize FastAPI app
app = FastAPI()

# Session middleware for cookie-backed sessions
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "your-own-secret"))

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Connect to DuckDB
con = duckdb.connect()

# Load the CSV file into a DuckDB table
print("Loading CSV file...")
con.execute("""
    CREATE TABLE IF NOT EXISTS supply_chain AS
    SELECT * FROM read_csv_auto('DataCoSupplyChainDataset_UTF8.csv')
""")
print("CSV file loaded successfully!")

# Create tables for users, sessions, chat history with DuckDB-compatible identity columns

con.execute("CREATE SEQUENCE IF NOT EXISTS user_id_seq START 1;")
con.execute("CREATE SEQUENCE IF NOT EXISTS chat_id_seq START 1;")

con.execute("""
CREATE TABLE IF NOT EXISTS users (
  id INTEGER PRIMARY KEY DEFAULT nextval('user_id_seq'),
  username VARCHAR UNIQUE,
  hashed_password VARCHAR
)
""")
con.execute("""
CREATE TABLE IF NOT EXISTS sessions (
  session_id VARCHAR PRIMARY KEY,
  user_id INTEGER REFERENCES users(id),
  created_at TIMESTAMP DEFAULT NOW()
)
""")
con.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
  id INTEGER PRIMARY KEY DEFAULT nextval('chat_id_seq'),
  session_id VARCHAR REFERENCES sessions(session_id),
  is_user BOOLEAN,
  message TEXT,
  timestamp TIMESTAMP DEFAULT NOW()
)
""")


# Retrieve column names and data types
schema_info = con.execute("PRAGMA table_info('supply_chain')").fetchall()
columns_info = [f"{col[1]} ({col[2]})" for col in schema_info]

# Setup Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Password hashing utilities
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(plain: str) -> str:
    return pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_ctx.verify(plain, hashed)

# Initialize Qdrant client and embedding model
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
qdrant = QdrantVectorStore(
    client=qdrant_client,
    collection_name="docs_collection_2",
    embedding=embedding_model
)

# Routes for registration and login
@app.get("/register", response_class=HTMLResponse)
def register_form(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register")
def register(request: Request, username: str = Form(...), password: str = Form(...)):
    hashed = hash_password(password)
    try:
        con.execute(
            "INSERT INTO users (username, hashed_password) VALUES (?, ?)",
            (username, hashed),
        )
    except Exception:
        raise HTTPException(400, "Username already exists")
    return RedirectResponse(url="/login", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
def login(request: Request, username: str = Form(...), password: str = Form(...)):
    row = con.execute(
        "SELECT id, hashed_password FROM users WHERE username = ?", (username,)
    ).fetchone()
    if not row or not verify_password(password, row[1]):
        raise HTTPException(401, "Invalid credentials")
    sess_id = str(uuid.uuid4())
    con.execute(
        "INSERT INTO sessions (session_id, user_id) VALUES (?, ?)",
        (sess_id, row[0]),
    )
    request.session["session_id"] = sess_id
    return RedirectResponse(url="/", status_code=303)

# Home (chat) page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    sess_id = request.session.get("session_id")
    if not sess_id:
        return RedirectResponse(url="/login")
    history = con.execute(
        "SELECT is_user, message FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT 10",
        (sess_id,)
    ).fetchall()
    msgs = [(bool(r[0]), r[1]) for r in reversed(history)]
    return templates.TemplateResponse("chat.html", {"request": request, "messages": msgs})

@app.post("/chat")
async def chat(request: Request, message: str = Form(...)):
    sess_id = request.session.get("session_id")
    if not sess_id:
        raise HTTPException(401, "Not logged in")
    # save user message
    con.execute(
        "INSERT INTO chat_history (session_id, is_user, message) VALUES (?, TRUE, ?)",
        (sess_id, message),
    )

    try:
        # Prepare schema string
        schema_str = "\n".join(columns_info)
        # Core prompt
        prompt = f"""
            You are an AI assistant designed to help users access information from both documents and a SQL database.  

            **Database Schema:**
            Table: supply_chain
            {schema_str}

            -- Date Parsing Note
            -- Columns "order date (DateOrders)" and "shipping date (DateOrders)"
            -- are stored as TEXT in the CSV in the format M/D/YYYY HH:MM (e.g. 1/31/2018 22:56).

            -- In DuckDB, convert both text columns to TIMESTAMPs:
            --   STRPTIME(text_col, '%m/%d/%Y %H:%M') returns a TIMESTAMP.

            -- If you need Julian Day numbers, call julian() on a TIMESTAMP:
            --   julian(STRPTIME(text_col, '%m/%d/%Y %H:%M'))

            -- To compute a day-difference directly, use date_diff('day', ts1, ts2).

            -- Example usage in a SELECT:
            SELECT
            STRPTIME("order date (DateOrders)",  '%m/%d/%Y %H:%M') AS order_ts,
            STRPTIME("shipping date (DateOrders)", '%m/%d/%Y %H:%M') AS ship_ts,
            -- Julian Day numbers, if you need them:
            julian(order_ts) AS order_jd,
            julian(ship_ts)  AS ship_jd,
            -- Or simply diff in whole days:
            date_diff('day', order_ts, ship_ts) AS transit_days
            FROM your_table;

            -- To filter for “last year” based on either timestamp:
            WHERE
            STRPTIME("order date (DateOrders)",  '%m/%d/%Y %H:%M')
                >= (CAST(CURRENT_DATE AS TIMESTAMP) - INTERVAL '1 year')
            AND
            STRPTIME("shipping date (DateOrders)", '%m/%d/%Y %H:%M')
                >= (CAST(CURRENT_DATE AS TIMESTAMP) - INTERVAL '1 year');


            Note: Column names may contain spaces. When generating SQL queries, ensure that such column names are enclosed in double quotes. For example, use `"Order Item Profit Ratio"` instead of `Order_Item_Profit_Ratio`.

            When a user asks a question, determine the appropriate action:
            - **"sql"**: questions fully answered by SQL.
            - **"doc_retrieval"**: questions fully answered by documents.
            - **"both"**: questions needing both data and documents. Return **"doc_question"** for targeted retrieval.
            - **"none"**: casual conversation.

            Your response must be a JSON object with:
            1. **"response"**: A friendly, actionable answer.
            2. **"process"**: One of "doc_retrieval", "sql", "both", or "none".
            3. **"sql"**: If process is "sql" or "both", exact SQL; else null.
            4. **"doc_question"**: If process is "both", concise query for documents; else null.

            User input: "{message}"
        """
        decision = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        raw = decision.text.strip()
        start, end = raw.find('{'), raw.rfind('}')+1
        result = json.loads(raw[start:end])
        process = result.get("process", "none")

        if process == "doc_retrieval":
            print("Doc retrieval")
            docs = qdrant.similarity_search(message, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            followup_prompt = f"""
                You're a chatbot AI assistant.
                Using the following document context, provide a **detailed, business-focused analysis** that:
                1. Answers the user's question fully.
                2. Explains implications and potential impacts.
                3. Offers actionable recommendations.

                Context:
                {context}

                User Question:
                {message}
            """
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=followup_prompt)
            print("=== DOC FOLLOWUP PROMPT ===")
            print(followup_prompt)
            print("=== DOC FOLLOWUP RESPONSE ===")
            print(resp.text.strip())
            return {"response": resp.text.strip()}

        elif process == "sql":
            print("SQL needed")
            sql_query = result.get("sql")
            print("=== SQL QUERY ===")
            print(sql_query)
            rows = con.execute(sql_query).fetchall()
            cols = [d[0] for d in con.description]
            data = [dict(zip(cols, r)) for r in rows]
            followup_prompt = f"""
                You're a chatbot AI assistant.
                Given this table of results, provide a **comprehensive business analysis** that:
                1. Interprets the data points clearly.
                2. Highlights key trends and insights.
                3. Recommends strategic actions and next steps.

                Data:
                {data}

                Original Question:
                {message}
            """
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=followup_prompt)
            print("=== SQL FOLLOWUP PROMPT ===")
            print(followup_prompt)
            print("=== SQL FOLLOWUP RESPONSE ===")
            print(resp.text.strip())
            return {"response": resp.text.strip()}

        elif process == "both":
            print("Both needed")
            doc_q = result.get("doc_question", message)
            docs = qdrant.similarity_search(doc_q, k=3)
            context = "\n\n".join([d.page_content for d in docs])
            hybrid_prompt = f"""
                You have two sources:
                1) Context from documents:
                {context}

                2) The database schema for table supply_chain:
                {schema_str}

                -- Date Parsing Note
                -- Columns "order date (DateOrders)" and "shipping date (DateOrders)"
                -- are stored as TEXT in the CSV in the format M/D/YYYY HH:MM (e.g. 1/31/2018 22:56).

                -- In DuckDB, convert both text columns to TIMESTAMPs:
                --   STRPTIME(text_col, '%m/%d/%Y %H:%M') returns a TIMESTAMP.

                -- If you need Julian Day numbers, call julian() on a TIMESTAMP:
                --   julian(STRPTIME(text_col, '%m/%d/%Y %H:%M'))

                -- To compute a day-difference directly, use date_diff('day', ts1, ts2).

                -- Example usage in a SELECT:
                SELECT
                STRPTIME("order date (DateOrders)",  '%m/%d/%Y %H:%M') AS order_ts,
                STRPTIME("shipping date (DateOrders)", '%m/%d/%Y %H:%M') AS ship_ts,
                -- Julian Day numbers, if you need them:
                julian(order_ts) AS order_jd,
                julian(ship_ts)  AS ship_jd,
                -- Or simply diff in whole days:
                date_diff('day', order_ts, ship_ts) AS transit_days
                FROM your_table;

                -- To filter for “last year” based on either timestamp:
                WHERE
                STRPTIME("order date (DateOrders)",  '%m/%d/%Y %H:%M')
                    >= (CAST(CURRENT_DATE AS TIMESTAMP) - INTERVAL '1 year')
                AND
                STRPTIME("shipping date (DateOrders)", '%m/%d/%Y %H:%M')
                    >= (CAST(CURRENT_DATE AS TIMESTAMP) - INTERVAL '1 year');


                Note: Column names may contain spaces. When generating SQL queries, ensure that such column names are enclosed in double quotes. For example, use "Order Item Profit Ratio" instead of Order_Item_Profit_Ratio.

                Based on both sources and these rules, produce a JSON object with:
                - "response": a brief acknowledgement like "Running SQL now..."  
                - "process": "sql"  
                - "sql": the exact SQL query to answer the question

                Question:
                {message}
                """

            hybrid = client.models.generate_content(model="gemini-2.0-flash", contents=hybrid_prompt)
            print("=== HYBRID PROMPT ===")
            print(hybrid_prompt)
            print("=== HYBRID RESPONSE ===")
            print(hybrid.text.strip())
            hj = json.loads(hybrid.text.strip()[hybrid.text.find('{'):hybrid.text.rfind('}')+1])
            sql_query = hj.get("sql")
            print("=== HYBRID SQL QUERY ===")
            print(sql_query)
            rows = con.execute(sql_query).fetchall()
            cols = [d[0] for d in con.description]
            data = [dict(zip(cols, r)) for r in rows]
            followup_prompt = f"""
                You're a chatbot AI assistant.
                Using both the retrieved documents and the SQL result below, craft a **detailed business report** that:
                1. Answers the user's original question: {message}
                2. Explains data-driven insights and document-based context.
                3. Provides clear, actionable recommendations.

                Document Context:
                {context}

                SQL Data:
                {data}
            """
            resp = client.models.generate_content(model="gemini-2.0-flash", contents=followup_prompt)
            print("=== FINAL REPORT PROMPT ===")
            print(followup_prompt)
            print("=== FINAL REPORT RESPONSE ===")
            print(resp.text.strip())
            return {"response": resp.text.strip()}

        # process == none or unexpected
        return {"response": result.get("response", "")}

    except Exception as e:
        return {"response": f"Sorry, there was an error: {str(e)}"}
