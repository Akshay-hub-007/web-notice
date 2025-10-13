from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
import os
from langchain_google_genai import ChatGoogleGenerativeAI
load_dotenv()

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

persist_dir = "./chroma_langchain_db"

if os.path.exists(persist_dir) and os.listdir(persist_dir):
    print("Existing vector store found.")
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
else:
    print("📄 No existing vector store found. Creating one now...")

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



question = "how to create notice"
results = vector_store.similarity_search(query=question, k=4)


# Print and collect chunks
similar_chunks_list = []
for i, doc in enumerate(results, 1):
    print(f"{i}. {doc.page_content}\n")
    similar_chunks_list.append(doc.page_content)

# Combine all chunks into one string
similar_chunks = "\n\n".join(similar_chunks_list)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

prompt = f"""
You are an intelligent assistant that answers user questions based only on the given document context.

Context:
\"\"\"{similar_chunks}\"\"\"

Question:
{question}

Instructions:
- Use only the information from the context above.
- If the answer is not clearly found in the context, say "The document does not provide this information."
- Provide a clear and concise answer in a professional tone.
"""

res = llm.invoke(prompt)

