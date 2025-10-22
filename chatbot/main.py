from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_google_genai import GoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langgraph.graph import StateGraph
from pydantic import BaseModel
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import create_agent
import os
from langgraph.graph import START , END
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# llm1 = HuggingFaceEndpoint(
#     repo_id="microsoft/Phi-3-mini-4k-instruct",
#     task="text-generation",
#     max_new_tokens=512,
#     do_sample=False,
#     repetition_penalty=1.03,
# )
# llm = ChatHuggingFace(llm=llm1, verbose=True)
# llm = ChatGroq(model="gemma2-9b-it")
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
graph = StateGraph(dict)

db = SQLDatabase.from_uri("mysql+pymysql://root:1234@localhost:3306/web-notice")

model = llm

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

# def dbNode(state: dict):
#     """
#     Processes a user query, retrieves data from the database using SQLDatabaseToolkit,
#     and returns the agent's response.
#
#     Args:
#         state (dict): A dictionary containing the key 'query' with the user's question.
#
#     Returns:
#         dict: Contains the original query and the full response from the agent.
#     """
#     user_query = state.get('query', '')
#
#     system_prompt = (
#         "You are an assistant. Make database calls similar to the question, "
#         "retrieve documents, and provide an answer."
#     )
#
#     # Initialize the toolkit and retrieve tools
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)
#     tools = toolkit.get_tools()
#
#     # Create an agent with the system prompt
#     agent = create_agent(
#         model=model,
#         tools=tools,
#         system_prompt=system_prompt
#     )
#
#     # Prepare input messages
#     inputs = {"messages": [{"role": "user", "content": user_query}]}
#
#     # Invoke the agent
#     result = agent.invoke(inputs)
#     print(result)
#
#     return {"query": user_query, "response": result}
def dbNode(state: dict):
    user_query = state.get('query', '')
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    tools = toolkit.get_tools()
    system_prompt = """
     You are a helpful assistant specialized in answering database queries.
     - Always give responses in plain text only.
     - Do not return any JSON, code blocks, or markdown formatting.
     - If the query returns multiple rows, list them one after another, each on a new line.
     - Format each record as:
       Title: <title>, Content: <content>, Created At: <created_at>, Expiry Date: <expiry_date>
     - Do not include extra symbols like *, -, or quotes.
     - If no data matches, reply exactly with: No relevant data found.
     - Keep the tone polite and concise.
     """
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    result = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

    print(result)

    # Extract final AI message
    final_answer = ""

    # Method 1: Get the last message directly
    if result.get('messages'):
        last_message = result['messages'][-1]
        final_answer = last_message.content

    # Method 2: More explicit - check if it's an AIMessage
    from langchain_core.messages import AIMessage

    if result.get('messages'):
        for msg in reversed(result['messages']):
            if isinstance(msg, AIMessage):
                # Handle both dict and string outputs safely
                content = msg.content
                if isinstance(content, dict):
                    # Extract only the plain text part
                    final_answer = content.get("text", "")
                else:
                    final_answer = str(content)
                break

    print("Final Answer:", final_answer)
    return {"query": user_query, "response": final_answer}
def query_node(state: dict):
    user_query = state["query"]
    persist_dir = "./chroma_langchain_db"

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print("Existing vector store found.")
        vector_store = Chroma(
            collection_name="example_collection",
            embedding_function=embeddings,
            persist_directory=persist_dir,
        )
    else:
        print("No existing vector store found. Creating one now...")
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
    You are a knowledgeable assistant that answers questions based ONLY on the given context.
    - Provide concise and accurate answers.
    - If the answer is not found, reply: "The document does not provide this information."
    - Format lists or multiple points clearly.
    - Respond politely.

    Context:
    \"\"\"{context}\"\"\"

    Question:
    {user_query}
    """
    answer = llm.invoke(prompt)
    print("\nAnswer:\n", answer.content)
    return {"query": user_query, "answer": answer.content}

def custom_category(state:dict):
    pass

graph.add_node("classify_query", classify_query_node)
graph.add_node("db_node", dbNode)
graph.add_node("rag_node", query_node)
graph.add_node("custom_category",custom_category)
graph.add_edge(START , "classify_query")
graph.add_conditional_edges("classify_query",classify_category,{"db_node":"db_node","rag_node":"rag_node","custom_category":"custom_category"})
graph.add_edge("db_node",END)
graph.add_edge("rag_node",END)
graph.add_edge("custom_category",END)
user_input = {"query": "What is the maximum file size that can be attached?"}

# dbNode({"query" : "how many notices are there in notices db .what are they list them"})


workflow = graph.compile()

# res = workflow.invoke({"query": "how many notices are there in notices db .what are they list them"})

# print(res)