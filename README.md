# **PDF-Based Chatbot Using LangChain and HuggingFace**

This repository provides a framework for building a chatbot that answers questions from PDF documents using LangChain, FAISS, and HuggingFace. It combines information retrieval with generative AI to create a powerful Retrieval-Augmented Generation (RAG) system for context-driven responses.

---

## **Features**

1. PDF Document Parsing: Extracts text from PDFs and splits them into manageable chunks.  
2. Semantic Search: Uses FAISS and HuggingFace Embeddings for efficient and accurate information retrieval.  
3. Generative Responses: Combines retrieved information with a language model for precise, context-aware answers.  
4. Customizable Prompting: Define tailored prompts to adjust the chatbot's behavior for specific domains.  

---

## **Use Cases**

This framework is suitable for:  

- Document Q&A: Create chatbots for user manuals, legal documents, research papers, etc.  
- Customer Support: Automate responses using product documentation or FAQs.  
- Knowledge Management: Enable organizations to make internal documents easily searchable.  

---

## **Requirements**

To use this chatbot, you need the following:  

- Python 3.8 or later  
- The following Python libraries:  
  - `langchain`  
  - `faiss-cpu`  
  - `transformers`  
  - `PyMuPDF` (for PDF parsing)  

Install the dependencies using this command:  

```
pip install langchain faiss-cpu transformers PyMuPDF
```

---

## **How It Works**

1. **Document Loading**:  
   PDFs are parsed into text chunks using PyMuPDF.  
   Text is split into overlapping sections using RecursiveCharacterTextSplitter for better retrieval.  

2. **Vector Store Creation**:  
   Chunks are embedded using HuggingFace Embeddings.  
   FAISS indexes these embeddings for fast similarity searches.  

3. **RAG Chatbot**:  
   A generative language model (e.g., `facebook/opt-350m`) processes retrieved chunks to generate responses.  
   A customizable prompt ensures answers are concise and contextually relevant.  

4. **Interactive Chat**:  
   Users can interact with the chatbot to query the document.  

---

## **Usage**

1. Place your target PDF file (e.g., `your_document.pdf`) in the project directory.  

2. Update the file path in the script:  

```
file_path = "/path/to/your_document.pdf"
```

3. Run the script:  

```
python chatbot.py
```

4. Ask questions to the chatbot in the terminal. For example:  

```
You: What is the refund policy?
Chatbot: Refunds must be requested within 30 days of purchase, subject to a 10% processing fee.
```

Type `exit` to quit the session.  

---

## **File Structure**

- **`chatbot.py`**: Main script for document loading, vector store creation, and chatbot interaction.  
- **`requirements.txt`**: List of dependencies for the project.  

---

## **Customization**

- **Prompt Template**: Modify the prompt in the `build_chatbot` function to adjust the chatbot's behavior and tone.  
- **Embeddings and Models**: Replace `sentence-transformers/all-MiniLM-L6-v2` or `facebook/opt-350m` with other models for improved embeddings or generative quality.  

---

## **Known Limitations**

- **PDF Parsing**: Results depend on the quality and structure of the input PDF. Non-readable PDFs (scans) may require OCR preprocessing.  
- **Model Size**: Larger models yield better responses but may require significant computational resources.  
- **Domain Adaptation**: For specialized domains, consider fine-tuning embeddings or the language model.  

---

## **Contributing**

We welcome contributions! If you have suggestions or improvements, feel free to create an issue or submit a pull request.  
License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this software under the following conditions:

The original copyright notice and permission notice must be included in all copies or substantial portions of the software.
This software is provided "as-is," without warranty of any kind, express or implied.
For full details, see the LICENSE file included in this repository.
