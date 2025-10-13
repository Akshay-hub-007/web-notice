from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent
from langgraph.graph import StateGraph
from pydantic import BaseModel


load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")


graph = StateGraph(dict)


class QueryClassification(BaseModel):
    category: str

def classify_query_node(state: dict):
    user_query = state['query']

    classification_prompt = f"""
You are an intelligent query classifier for a chatbot system.
Your task is to determine whether the given user query is:
  1. related to the database (like CRUD operations on notices), or
  2️. a general question or documentation-related query.

Categories you can return:
- "db_calls" → queries related to MySQL tables such as 'notices', 'users', etc.
- "question" → generic, FAQ-like, or conceptual questions.

--- Example ---
Query: "Show all notices about exams" → db_calls
Query: "What is a notice board?" → question

Now classify the query below carefully.

Query: "{user_query}"
"""

    # Use structured output
    llm_with_classify = llm.with_structured_output(QueryClassification)
    result = llm_with_classify.invoke(classification_prompt)
    print("Classification Result:", result.category)
    return {'query' : user_query ,'category' : result.category}


def classify_categoty(state:dict):

    if state['category'] == 'db_calls':
        return 'db_node'
    elif state['category'] == 'question':
        return 'rag_node'
    else:
        return 'custom_category'

class SqlQuery(BaseModel):
    query :str
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
1. Generate a valid SQL query that fulfills the user's request.
2️. Always consider **fuzzy keyword variations**:
   - Handle plural/singular (e.g., "exam" ↔ "exams")
   - Handle case-insensitive matches.
   - Use LIKE with wildcards or LOWER() for safer search.
3️. Use proper SQL syntax for the fields above.
4️. Only return the SQL query (no explanations or comments).
5️. Assume the table name is exactly 'notices'.
6️. Do not return placeholders; use actual field names.

Examples:
- Query: "Show all important notices"
  SQL: SELECT * FROM notices WHERE priority = 'important';

- Query: "Get all active notices posted by user 5"
  SQL: SELECT * FROM notices WHERE is_active = TRUE AND posted_by = 5;

- Query: "Find exam related notices"
  SQL: SELECT * FROM notices WHERE LOWER(title) LIKE '%exam%' OR LOWER(content) LIKE '%exam%';

Now, generate the SQL query for the user query above.
"""
    llm_with_sql_query = llm.with_structured_output(SqlQuery)
    res = llm_with_sql_query.invoke(prompt)
    print(res.query)

graph.add_node("classify_query", classify_query_node)
graph.add_node("db_node" , dbNode)

user_input = {"query": "Show all notices about exams"}

dbNode(user_input)

