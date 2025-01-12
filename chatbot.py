from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import fitz

def load_document(file_path):
    """Load the PDF and split it into smaller chunks."""
    with fitz.open(file_path) as pdf:
        raw_documents = []
        for page in pdf:
            text = page.get_text()
            metadata = {"page_number": page.number + 1}
            raw_documents.append(Document(page_content=text, metadata=metadata))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Define chunk size for splitting
        chunk_overlap=100,  # Define overlap size
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    documents = text_splitter.split_documents(raw_documents)
    return documents

def create_vector_store(documents):
    """Create a FAISS vector store with the documents."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    vectorstore = FAISS.from_documents(
        documents,
        embeddings
    )
    return vectorstore

def build_chatbot(vectorstore):
    """Build the chatbot with retrieval-augmented generation."""
    model_name = "facebook/opt-350m"  # Define the language model to use
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    llm_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    llm = HuggingFacePipeline(
        pipeline=llm_pipeline,
        model_kwargs={"temperature": 0.7}
    )

    prompt_template = """Based on the following context from the document, please provide a clear and concise answer to the question. If the context doesn't contain enough information to answer the question, please say "I apologize, but I don't have enough information in the document to answer that question accurately."

Context: {context}

Question: {question}

Answer:"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}  # Retrieve top-k relevant chunks
        ),
        chain_type_kwargs={
            "prompt": PROMPT
        },
        return_source_documents=False  # Exclude source docs from response
    )

    return qa_chain

def process_response(response):
    """Clean and format the chatbot's response."""
    if isinstance(response, dict):
        response = response.get('result', '')

    response = str(response).strip()

    cleanup_phrases = [
        "Based on the context provided,",
        "According to the context,",
        "Based on the information provided,",
        "Context:",
        "Question:",
        "Answer:",
        "Concise answer:"
    ]

    for phrase in cleanup_phrases:
        if response.lower().startswith(phrase.lower()):
            response = response[len(phrase):].strip()

    if not response or response.isspace():
        return "I apologize, but I don't have enough information to answer that question."

    return response

def interact_with_chatbot(chatbot):
    """Interactive loop for user interaction."""
    print("\nWelcome to the PDF-Based Chatbot!")
    print("\nAsk questions based on the content of your document.")
    print("Type 'exit' to quit.")

    while True:
        try:
            query = input("\nYou: ").strip()
            if query.lower() == 'exit':
                print("Goodbye!")
                break
            if not query:
                print("Please enter a valid question.")
                continue

            response = chatbot.invoke(query)

            cleaned_response = process_response(response)

            print(f"\nChatbot: {cleaned_response}")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            print("Please try rephrasing your question or ask something else.")

if __name__ == "__main__":
    try:
        file_path = "path/to/your_document.pdf"

        print("Loading and processing document...")
        documents = load_document(file_path)

        print("Creating vector store...")
        vectorstore = create_vector_store(documents)

        print("Building chatbot...")
        chatbot = build_chatbot(vectorstore)

        print("\nInitialization complete!")
        interact_with_chatbot(chatbot)

    except Exception as e:
        print(f"Initialization error: {str(e)}")
