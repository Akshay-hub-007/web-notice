from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from pydantic import BaseModel
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
graph = StateGraph(dict)

class QueryClassification(BaseModel):
    category: str

class SqlQuery(BaseModel):
    query: str

def classify_query_node(state: dict):
    user_query = state['query']
    classification_prompt = f"""
You are an intelligent query classifier for a chatbot system.
Your task is to determine whether the given user query is:
  1. related to the database (like CRUD operations on notices), or
  2. a general question or documentation-related query.

Categories you can return:
- "db_calls" → queries related to MySQL tables such as 'notices', 'users', etc.
- "question" → generic, FAQ-like, or conceptual questions.

Query: "{user_query}"
"""
    llm_with_classify = llm.with_structured_output(QueryClassification)
    result = llm_with_classify.invoke(classification_prompt)
    print("Classification Result:", result.category)
    return {'query': user_query, 'category': result.category}

def classify_category(state: dict):
    if state['category'] == 'db_calls':
        return 'db_node'
    elif state['category'] == 'question':
        return 'rag_node'
    else:
        return 'custom_category'

def dbNode(state: dict):
    user_query = state['query']
    prompt = f"""
You are an expert SQL query generator.

User query: "{user_query}"

Database schema:
Table: notices
Fields:
- id (integer, primary key)
- title (varchar)
- content (text)
- priority (varchar: normal, important, urgent)
- attachment (file path, nullable)
- created_at (datetime)
- updated_at (datetime)
- expiry_date (date, nullable)
- is_active (boolean)
- posted_by (foreign key to users)

Task:
Generate a valid SQL query that fulfills the user's request.
Only return the SQL query string.
"""
    llm_with_sql_query = llm.with_structured_output(SqlQuery)
    res = llm_with_sql_query.invoke(prompt)
    print("\nGenerated SQL Query:\n", res.query)
    return {"query": user_query, "sql_query": res.query}

def query_node(state: dict):
    user_query = state["query"]
    persist_dir = "./chroma_langchain_db"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(" Existing vector store found.")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        print(" No existing vector store found. Creating one now...")
        file_path = "web_notice_info.pdf"
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=50
        )
        splitted_docs = text_splitter.split_documents(docs)
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
        vector_store.add_documents(splitted_docs)
    results = vector_store.similarity_search(query=user_query, k=4)
    context = "\n\n".join([doc.page_content for doc in results])
    prompt = f"""
You are an intelligent assistant that answers user questions based only on the given document context.

Context:
\"\"\"{context}\"\"\"

Question:
{user_query}

If the answer is not found in the context, say:
"The document does not provide this information."
"""
    answer = llm.invoke(prompt)
    print("\nAnswer:\n", answer.content)
    return {"query": user_query, "answer": answer.content}

graph.add_node("classify_query", classify_query_node)
graph.add_node("db_node", dbNode)
graph.add_node("rag_node", query_node)
graph.add_edge("classify_query", classify_category)

user_input = {"query": "What is the maximum file size that can be attached?"}
result = classify_query_node(user_input)
if result["category"] == "db_calls":
    dbNode(result)
else:
    query_node(result)
