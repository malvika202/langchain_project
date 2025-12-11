from langchain_ollama import ChatOllama 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser 

from langchain_community.document_loaders import PyPDFLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS 
from langchain_community.embeddings import OllamaEmbeddings 

# Load PDF 

pdf_path = "/workspaces/langchain_project/disney_movies_with_age_rating.pdf" 
loader = PyPDFLoader(pdf_path) 
documents = loader.load() 

# Split into chunks 

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200) 
docs = text_splitter.split_documents(documents) 

# Embed & store in vector database 

embeddings = OllamaEmbeddings(model="mxbai-embed-large") # or another embedding model 
vectorstore = FAISS.from_documents(docs, embeddings) 

# Create retriever 

retriever = vectorstore.as_retriever() 

 

# 5. Build model 

model = ChatOllama(model="recommendations") # or your custom model name 

 

# 6. Create prompt with context 

prompt = ChatPromptTemplate.from_template(""" 
You are a helpful AI that recommends and explains Disney movies. 
Use ONLY the following context from the PDF to answer the user question: 
{context} 

User question: {question} 
""") 

 

parser = StrOutputParser() 

 

# 7. Create retrieval-augmented chain 

def answer_question(query): 

# Retrieve relevant info 

relevant_docs = retriever.get_relevant_documents(query) 
context = "\n\n".join([d.page_content for d in relevant_docs]) 

# Run LLM 

chain = prompt | model | parser 
return chain.invoke({"context": context, "question": query}) 

# Example 

response = answer_question("What is the plot of Wish?") 
print(response) 