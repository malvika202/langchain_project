from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFOCRLoader

# Load PDF
loader = PyPDFOCRLoader("/workspaces/langchain_project/disney_movies_expanded.pdf")
docs = loader.load()

print(f"Number of pages loaded: {len(docs)}")
print("Preview first page:")
print(docs[0].page_content[:1000])


print(f"Number of pages loaded: {len(docs)}\n")
print("Preview first page:\n")
print(docs[0].page_content[:1000])  # first 1000 characters

# Split text
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=300)
chunks = text_splitter.split_documents(docs)
print(f"\nNumber of chunks created: {len(chunks)}")
print("Preview first chunk:\n")
print(chunks[0].page_content[:500])

# Embed & create vector store
embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

# Test a query
query = "PG Disney movies from 2023"
relevant_docs = retriever.get_relevant_documents(query)
print(f"\nNumber of retrieved chunks for query: {len(relevant_docs)}")
for i, doc in enumerate(relevant_docs):
    print(f"\n--- Chunk {i+1} ---\n")
    print(doc.page_content[:500])
