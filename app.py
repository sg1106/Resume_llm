

import os
import PyPDF2
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai
from chromadb.utils import embedding_functions
import logging
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import warnings
from dotenv import load_dotenv
import os

# Suppress resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
PDF_PATH = "swastik_resume_ats.pdf"  # Hardcoded PDF file path
RESUME_LINK = "https://drive.google.com/file/d/1SRjVRVYyLxjWtMa4t_4dToJIS3QD7ZHe/view?usp=drive_link"
collection = None
model = None
history = []


load_dotenv()




# Configure Gemini API
def configure_gemini():
    api_key = os.getenv("API_KEY")  # Replace with your actual Gemini API key
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        logger.error(f"Failed to configure Gemini API: {e}")
        raise

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page_num, page in enumerate(reader.pages, 1):
                page_text = page.extract_text() or ""
                text += f"\n[Page {page_num}]\n{page_text}\n"
            if not text.strip():
                raise ValueError("No text could be extracted from the PDF")
            logger.info(f"Successfully extracted text from {pdf_path}")
            return text
    except Exception as e:
        logger.error(f"Error reading PDF {pdf_path}: {e}")
        raise

# Split text into chunks
def chunk_text(text, chunk_size=500, overlap=50):
    try:
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        for i, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
                if i > 0 and overlap > 0:
                    prev_chunk = chunks[-1].split('. ')[-1] if chunks else ""
                    current_chunk = prev_chunk + ". " + current_chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks
    except Exception as e:
        logger.error(f"Error chunking text: {e}")
        raise

# Store in vector database
def store_in_vector_db(chunks, db_path="chroma_db"):
    try:
        client = chromadb.PersistentClient(path=db_path)
        embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")
        collection_name = "pdf_collection"
        
        try:
            client.delete_collection(collection_name)
        except:
            pass
        
        collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
        
        for i, chunk in enumerate(chunks):
            collection.add(
                documents=[chunk],
                ids=[f"chunk_{i}_{uuid.uuid4()}"],
                metadatas=[{"chunk_index": i, "timestamp": datetime.now().isoformat()}]
            )
        logger.info(f"Stored {len(chunks)} chunks in vector database")
        return collection
    except Exception as e:
        logger.error(f"Error storing in vector database: {e}")
        raise

# Retrieve relevant chunks
def retrieve_relevant_chunks(query, collection, top_k=10):
    try:
        results = collection.query(query_texts=[query], n_results=top_k)
        chunks = results['documents'][0]
        logger.info(f"Retrieved {len(chunks)} relevant chunks for query: {query}")
        return chunks
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise

# Generate answer using Gemini
def generate_answer(query, context, history, model):
    try:
        history = history[-5:]
        history_text = ""
        for turn in history:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

        prompt = f"""You are an expert assistant tasked with answering questions based solely on the provided PDF document context.
Instructions:
- Provide a **complete**, **accurate**, and **detailed** response.
- Use **bullet points** or **numbered lists** for clarity.
- Include **section titles** and **subsection titles** where applicable.
- If the query asks about experience or internships, include a subsection titled "Internship Details" or "Professional Experience".
- If the context lacks information to answer the query, state: "The document does not contain that information."
- Avoid speculative answers or external knowledge.
- Ensure the response is well-structured and concise.

Conversation History:
{history_text}

Context from PDF:
{context}

Current User Question: {query}

Response:
"""

        response = model.generate_content(prompt)
        answer = response.text.strip()
        # Append the reference line with the resume link as a clickable HTML anchor tag in blue
        answer += f'\n\nFor reference, please click on the link below to view the resume: <a href="{RESUME_LINK}" target="_blank" style="color: blue;">View Resume</a>'
        logger.info(f"Generated answer for query: {query}")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise

# Initialize RAG pipeline on startup
def init_rag_pipeline():
    global collection, model, history
    try:
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        collection = store_in_vector_db(chunks)
        model = configure_gemini()
        history = []
        logger.info("RAG pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        raise

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    global collection, model, history
    try:
        data = request.get_json()
        query = data.get('question', '').strip()
        
        if not query:
            return jsonify({'error': 'No question provided'}), 400
        if not collection or not model:
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
            
        relevant_chunks = retrieve_relevant_chunks(query, collection)
        context = "\n".join(relevant_chunks)
        answer = generate_answer(query, context, history, model)
        
        history.append({"question": query, "answer": answer})
        
        return jsonify({'answer': answer}), 200
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_rag_pipeline()  # Initialize pipeline on startup
    app.run(debug=True)