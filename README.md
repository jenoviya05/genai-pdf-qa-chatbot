## Development of a PDF-Based Question-Answering Chatbot Using LangChain

### AIM:
To design and implement a question-answering chatbot capable of processing and extracting information from a provided PDF document using LangChain, and to evaluate its effectiveness by testing its responses to diverse queries derived from the document's content.

### PROBLEM STATEMENT:

To develop an intelligent chatbot that reads and processes PDF documents, retrieves relevant information based on user queries, and generates accurate answers using LangChain. The system aims to reduce manual searching and provide quick access to information from PDF files.

### DESIGN STEPS:

### STEP 1: PDF Loading and Text Extraction
- Import the required LangChain libraries.
- Load the PDF document using PyPDFLoader.
- Extract text from all pages of the PDF document.
### STEP 2: Text Splitting and Embedding
- Split the extracted text into smaller chunks using RecursiveCharacterTextSplitter.
- Convert the text chunks into vector embeddings.
- Store the embeddings in a vector database using FAISS for efficient similarity searching.
### STEP 3: Question Answering and Response Generation
- Accept questions from the user.
- Retrieve relevant text chunks from the vector database using similarity search.
- Pass the retrieved context and user question to the language model.
- Generate and display relevant answers based on the PDF content.

### PROGRAM:

~~~
%pip install "langchain==0.2.17" "langchain-community==0.2.19" "langchain-core==0.2.43" "langchain-text-splitters==0.2.4" "transformers==4.44.2" "sentence-transformers==3.0.1" "faiss-cpu==1.8.0.post1" "pypdf==4.3.1" torch

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from transformers import pipeline

print("All libraries imported successfully.")

PDF_PATH = "sample_knowledge_document.pdf"  # Change this if using another PDF

if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
        "PDF not found. Keep sample_knowledge_document.pdf in the same folder "
        "as this notebook, or update PDF_PATH with the correct file path."
    )

print("PDF found:", PDF_PATH)

loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print("PDF loaded successfully.")
print("Number of pages:", len(documents))
print("\nText from the first page:")
print(documents[0].page_content[:1000] if documents else "No text extracted.")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = text_splitter.split_documents(documents)

print("Number of text chunks:", len(chunks))
print("\nFirst chunk:")
print(chunks[0].page_content if chunks else "No chunks created.")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("Embedding model loaded.")

vectorstore = FAISS.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("FAISS vector database created.")

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)
print("Retriever created. It will search for the 3 most relevant text chunks.")

text_generation_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-small",
    tokenizer="google/flan-t5-small",
    max_new_tokens=120,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("Local language model loaded.")

qa_prompt = PromptTemplate.from_template(
    """Answer the question using only the information in the context.
If the answer is not available in the context, say:
"I could not find this information in the provided PDF."

Context:
{context}

Question: {input}

Answer:"""
)
print("Question-answering prompt created.")

document_chain = create_stuff_documents_chain(
    llm=llm,
    prompt=qa_prompt
)

qa_chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=document_chain
)

print("PDF question-answering chatbot is ready.")

question = "What is cybersecurity?"
answer_result = qa_chain.invoke({"input": question})

print("Question:", question)
print("Answer:", answer_result["answer"])

test_questions = [
    "What is artificial intelligence?",
    "List some cybersecurity safety practices.",
    "What are the benefits of cloud computing?",
    "What does responsible use of technology mean?"
]

for number, question in enumerate(test_questions, start=1):
    answer_result = qa_chain.invoke({"input": question})
    print("\nTEST", number)
    print("Question:", question)
    print("Answer:", answer_result["answer"])

question = "What are the benefits of cloud computing?"
answer_result = qa_chain.invoke({"input": question})

print("Answer:", answer_result["answer"])
print("\nRetrieved PDF pages:")
for doc in answer_result["context"]:
    page_number = doc.metadata.get("page", "Unknown")
    if isinstance(page_number, int):
        page_number += 1  # Convert zero-based page index to page number
    print("\nPage:", page_number)
    print(doc.page_content)

while True:
    question = input("Ask something about the PDF (type exit to stop): ").strip()

    if question.lower() == "exit":
        print("Chat ended.")
        break

    if not question:
        print("Please type a question.")
        continue

    answer_result = qa_chain.invoke({"input": question})
    print("\nChatbot:", answer_result["answer"])


~~~

### OUTPUT:
<img width="958" height="537" alt="Screenshot 2026-11-06 091114" src="https://github.com/user-attachments/assets/014efc3b-b80c-4cc3-9253-047ec0b66016" />

<img width="937" height="751" alt="Screenshot 2026-11-06 091127" src="https://github.com/user-attachments/assets/cd14ef39-40e1-4cd2-933c-e06ac554e329" />

<img width="986" height="822" alt="Screenshot 2026-11-06 091144" src="https://github.com/user-attachments/assets/013e2260-e98c-4104-8bf5-548710987d1d" />

<img width="935" height="701" alt="Screenshot 2026-11-06 091152" src="https://github.com/user-attachments/assets/f58f09d1-297c-46c6-b538-d340fa600198" />

<img width="952" height="887" alt="Screenshot 2026-11-06 091211" src="https://github.com/user-attachments/assets/b25f7ee4-4558-43cf-9c13-0b147ab2b26a" />

<img width="892" height="672" alt="Screenshot 2026-11-06 091220" src="https://github.com/user-attachments/assets/47691f20-6d51-4719-8afb-c42d097fbbf2" />

<img width="931" height="897" alt="Screenshot 2026-11-06 091236" src="https://github.com/user-attachments/assets/88b16571-52ed-4039-b4fc-b6775c412a56" />

<img width="902" height="785" alt="Screenshot 2026-11-06 091244" src="https://github.com/user-attachments/assets/a87b73c0-be01-4b8a-82f3-5efe6595747a" />

<img width="720" height="912" alt="Screenshot 2026-11-06 091302" src="https://github.com/user-attachments/assets/7928edbc-735c-49cc-9078-951124ca88fc" />

<img width="1065" height="600" alt="Screenshot 2026-11-06 091316" src="https://github.com/user-attachments/assets/2c1f79c1-9470-4ffe-9a68-f48bee02fadf" />


### RESULT:
Thus, the PDF-based Question-Answering Chatbot was successfully designed and implemented using LangChain. The chatbot loads PDF documents, extracts and processes text, retrieves relevant information, and generates appropriate answers to user queries based on the document's content.
