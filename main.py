from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFOCRLoader


# Loads the PDF by converting it to images and extracts text 
# loader.load reads the pdf

loader = PyPDFOCRLoader("/workspaces/langchain_project/disney_movies_expanded.pdf")
docs = loader.load()

# We got a library (code made from other people) that helps us split text into chunks
# LLMS have context limits
# Number of characters, how many characters they overlap

from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# numerical vector - semantic representation of text
# vector for each chunk

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":5})

# AI was used
# Test retrieval
query = "PG Disney movies from 2023"
docs = retriever.get_relevant_documents(query)
for doc in docs:
    print(doc.page_content[:500])

# Retriever - LLM asks for relevant documents based on a query (k nearest neighbors)
# chunks stored in vectorstore

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Loads the model with name and parameters

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0.0, 
    top_p=0.9,
    num_predict=300
)

# AI was used
# Prompt template with context and question
# Needed to be strict about answering only from the context

prompt = ChatPromptTemplate.from_template("""
Context:
{context}

User Request:
{question}

Answer strictly using the Context above. 
If the answer is not in the Context, respond exactly:
"❌ The provided PDF does not contain this information."
""")

parser = StrOutputParser()


# AI was used
# uses retriever, hard fail if nothing found, connects everything, more grounding, debug (made by AI)

def get_recommendations(query):
    relevant_docs = retriever.get_relevant_documents(query)

    # HARD FAIL if nothing relevant
    if not relevant_docs:
        return "❌ No relevant information found in the PDF."

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # more grounding check
    if len(context.strip()) < 200:
        return "❌ The PDF does not contain enough information to answer this question."

    # DEBUG: print retrieved context (optional)
    # print("\n--- Retrieved Context ---")
    # for i, doc in enumerate(relevant_docs):
    #     print(f"Chunk {i+1}:\n{doc.page_content[:400]}\n")

    # Run the model
    # Parser makes sure output is string

    chain = prompt | model | parser
    return chain.invoke({"context": context, "question": query})


# AI was used to help tweak it
# Instructions - flush=True makes it print as soon as it starts
# EOF allows it to leave gracefully

def chat_loop():
    print("🎬 Disney Movie Recommendation Assistant", flush=True)
    print("Ask something like: 'Movies for age 7' or 'List PG movies'", flush=True)
    print("Type 'quit' to exit.\n", flush=True)

    while True:
        try:
            user_input = input("➡ Enter your question: ")
        except EOFError:
            # Works in non-interactive environments
            break

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye! 👋", flush=True)
            break

        answer = get_recommendations(user_input)
        print("\nRecommended Movies:\n", flush=True)
        print(answer, flush=True)
        print("\n" + "-"*60 + "\n", flush=True)

# Run the chat loop
if __name__ == "__main__":
    chat_loop()

