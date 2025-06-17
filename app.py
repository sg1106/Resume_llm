

# import os
# import PyPDF2
# from sentence_transformers import SentenceTransformer
# import chromadb
# import google.generativeai as genai
# from chromadb.utils import embedding_functions
# import logging
# import uuid
# from datetime import datetime
# from flask import Flask, request, render_template, jsonify
# import warnings
# from dotenv import load_dotenv
# import os

# # Suppress resource_tracker warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# # Set up logging
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# app = Flask(__name__)

# # Configuration
# PDF_PATH = "swastik_resume.pdf"  # Hardcoded PDF file path
# RESUME_LINK = "https://drive.google.com/file/d/1SRjVRVYyLxjWtMa4t_4dToJIS3QD7ZHe/view?usp=drive_link"
# collection = None
# model = None
# history = []


# load_dotenv()




# # Configure Gemini API
# def configure_gemini():
#     api_key = os.getenv("API_KEY")  # Replace with your actual Gemini API key
#     try:
#         genai.configure(api_key=api_key)
#         return genai.GenerativeModel('gemini-2.0-flash')
#     except Exception as e:
#         logger.error(f"Failed to configure Gemini API: {e}")
#         raise

# # Extract text from PDF
# def extract_text_from_pdf(pdf_path):
#     try:
#         if not os.path.exists(pdf_path):
#             raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
#         with open(pdf_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             text = ""
#             for page_num, page in enumerate(reader.pages, 1):
#                 page_text = page.extract_text() or ""
#                 text += f"\n[Page {page_num}]\n{page_text}\n"
#             if not text.strip():
#                 raise ValueError("No text could be extracted from the PDF")
#             logger.info(f"Successfully extracted text from {pdf_path}")
#             return text
#     except Exception as e:
#         logger.error(f"Error reading PDF {pdf_path}: {e}")
#         raise

# # Split text into chunks
# def chunk_text(text, chunk_size=500, overlap=50):
#     try:
#         sentences = text.split('. ')
#         chunks = []
#         current_chunk = ""
#         for i, sentence in enumerate(sentences):
#             sentence = sentence.strip()
#             if not sentence:
#                 continue
#             if len(current_chunk) + len(sentence) < chunk_size:
#                 current_chunk += sentence + ". "
#             else:
#                 if current_chunk:
#                     chunks.append(current_chunk.strip())
#                 current_chunk = sentence + ". "
#                 if i > 0 and overlap > 0:
#                     prev_chunk = chunks[-1].split('. ')[-1] if chunks else ""
#                     current_chunk = prev_chunk + ". " + current_chunk
#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())
#         logger.info(f"Created {len(chunks)} chunks from text")
#         return chunks
#     except Exception as e:
#         logger.error(f"Error chunking text: {e}")
#         raise

# # Store in vector database
# def store_in_vector_db(chunks, db_path="chroma_db"):
#     try:
#         client = chromadb.PersistentClient(path=db_path)
#         embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")
#         collection_name = "pdf_collection"
        
#         try:
#             client.delete_collection(collection_name)
#         except:
#             pass
        
#         collection = client.create_collection(name=collection_name, embedding_function=embedding_function)
        
#         for i, chunk in enumerate(chunks):
#             collection.add(
#                 documents=[chunk],
#                 ids=[f"chunk_{i}_{uuid.uuid4()}"],
#                 metadatas=[{"chunk_index": i, "timestamp": datetime.now().isoformat()}]
#             )
#         logger.info(f"Stored {len(chunks)} chunks in vector database")
#         return collection
#     except Exception as e:
#         logger.error(f"Error storing in vector database: {e}")
#         raise

# # Retrieve relevant chunks
# def retrieve_relevant_chunks(query, collection, top_k=10):
#     try:
#         results = collection.query(query_texts=[query], n_results=top_k)
#         chunks = results['documents'][0]
#         logger.info(f"Retrieved {len(chunks)} relevant chunks for query: {query}")
#         return chunks
#     except Exception as e:
#         logger.error(f"Error retrieving chunks: {e}")
#         raise

# # Generate answer using Gemini
# def generate_answer(query, context, history, model):
#     try:
#         history = history[-5:]
#         history_text = ""
#         for turn in history:
#             history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

#         prompt = f"""You are an expert assistant tasked with answering questions based solely on the provided PDF document context.
# Instructions:
# - Provide a **complete**, **accurate**, and **detailed** response.
# - Use **bullet points** or **numbered lists** for clarity.
# - Include **section titles** and **subsection titles** where applicable.
# - If the query asks about experience or internships, include a subsection titled "Internship Details" or "Professional Experience".
# - If the context lacks information to answer the query, state: "The document does not contain that information."
# - Avoid speculative answers or external knowledge.
# - Ensure the response is well-structured and concise.

# Conversation History:
# {history_text}

# Context from PDF:
# {context}

# Current User Question: {query}

# Response:
# """

#         response = model.generate_content(prompt)
#         answer = response.text.strip()
#         # Append the reference line with the resume link as a clickable HTML anchor tag in blue
#         answer += f'\n\nFor reference, please click on the link below to view the resume: <a href="{RESUME_LINK}" target="_blank" style="color: blue;">View Resume</a>'
#         logger.info(f"Generated answer for query: {query}")
#         return answer
#     except Exception as e:
#         logger.error(f"Error generating answer: {e}")
#         raise

# # Initialize RAG pipeline on startup
# def init_rag_pipeline():
#     global collection, model, history
#     try:
#         text = extract_text_from_pdf(PDF_PATH)
#         chunks = chunk_text(text)
#         collection = store_in_vector_db(chunks)
#         model = configure_gemini()
#         history = []
#         logger.info("RAG pipeline initialized successfully")
#     except Exception as e:
#         logger.error(f"Pipeline initialization error: {e}")
#         raise

# # Flask routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global collection, model, history
#     try:
#         data = request.get_json()
#         query = data.get('question', '').strip()
        
#         if not query:
#             return jsonify({'error': 'No question provided'}), 400
#         if not collection or not model:
#             return jsonify({'error': 'RAG pipeline not initialized'}), 500
            
#         relevant_chunks = retrieve_relevant_chunks(query, collection)
#         context = "\n".join(relevant_chunks)
#         answer = generate_answer(query, context, history, model)
        
#         history.append({"question": query, "answer": answer})
        
#         return jsonify({'answer': answer}), 200
#     except Exception as e:
#         logger.error(f"Error answering question: {e}")
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     init_rag_pipeline()  # Initialize pipeline on startup
#     app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))




import os
import PyPDF2
from sentence_transformers import SentenceTransformer
# import chromadb
import google.generativeai as genai
# from chromadb.utils import embedding_functions
import logging
import uuid
from datetime import datetime
from flask import Flask, request, render_template, jsonify
import warnings
from dotenv import load_dotenv
import os
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# Suppress resource_tracker warnings
warnings.filterwarnings("ignore", category=UserWarning, module="multiprocessing.resource_tracker")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
PDF_PATH = "swastik_resume.pdf"  # Hardcoded PDF file path
RESUME_LINK = "https://drive.google.com/file/d/1SRjVRVYyLxjWtMa4t_4dToJIS3QD7ZHe/view?usp=drive_link"
# EMBEDDING_MODEL_NAME = "all-MiniLM-L12-v2"
PICKLE_DB_PATH = "vector_data.pkl"
collection = None
model = None
history = []


load_dotenv()



# RESUME_LINK = "https://example.com/resume.pdf"  # Replace with actual resume link

# Logger setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini
def configure_gemini():
    api_key = os.getenv("API_KEY")
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

# Store chunks and embeddings using pickle
def store_in_pickle_db(chunks, model, pickle_path=PICKLE_DB_PATH):
    try:
        embeddings = model.encode(chunks)
        with open(pickle_path, 'wb') as f:
            pickle.dump({'chunks': chunks, 'embeddings': embeddings}, f)
        logger.info(f"Stored {len(chunks)} chunks and embeddings to {pickle_path}")
    except Exception as e:
        logger.error(f"Error storing pickle DB: {e}")
        raise

# Retrieve relevant chunks using cosine similarity
def retrieve_relevant_chunks(query, model, pickle_path=PICKLE_DB_PATH, top_k=10):
    if model is None:
        raise ValueError("Embedding model is not initialized. Call init_rag_pipeline() first.")
    try:
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        chunks = data['chunks']
        embeddings = data['embeddings']
        query_embedding = model.encode([query])
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [chunks[i] for i in top_indices]
        logger.info(f"Retrieved {len(results)} relevant chunks for query: {query}")
        return results
    except Exception as e:
        logger.error(f"Error retrieving chunks: {e}")
        raise

# Generate answer using Gemini
def generate_answer(query, context_chunks, history, model):
    try:
        history = history[-5:]
        history_text = ""
        for turn in history:
            history_text += f"User: {turn['question']}\nAssistant: {turn['answer']}\n"

        context = "\n\n".join(context_chunks)
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
        answer += f'\n\nFor reference, please click on the link below to view the resume: <a href="{RESUME_LINK}" target="_blank" style="color: blue;">View Resume</a>'
        logger.info(f"Generated answer for query: {query}")
        return answer
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise

# Initialize RAG pipeline on startup
def init_rag_pipeline():
    global rag_model, gemini_model, history
    try:
        logger.info("Starting RAG pipeline initialization...")
        
        if not os.path.exists(PDF_PATH):
            logger.error(f"PDF file does not exist at path: {PDF_PATH}")
            return
        
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        
        rag_model = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")
        # SentenceTransformer(EMBEDDING_MODEL_NAME)
        
        logger.info("Embedding model loaded successfully.")

        store_in_pickle_db(chunks, rag_model)
        gemini_model = configure_gemini()
        history = []
        
        logger.info("RAG pipeline initialized successfully.")
    except Exception as e:
        logger.error(f"Pipeline initialization error: {e}")
        rag_model = None  # explicitly clear it
        raise



# Flask routes
# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask_question():
#     global collection, model, history
#     try:
#         data = request.get_json()
#         query = data.get('question', '').strip()
        
#         if not query:
#             return jsonify({'error': 'No question provided'}), 400
#         # if not collection or not model:
#         #     return jsonify({'error': 'RAG pipeline not initialized'}), 500
            
#         relevant_chunks = retrieve_relevant_chunks(query, collection)
#         context = "\n".join(relevant_chunks)
#         answer = generate_answer(query, context, history, model)
        
#         history.append({"question": query, "answer": answer})
        
#         return jsonify({'answer': answer}), 200
#     except Exception as e:
#         logger.error(f"Error answering question: {e}")
#         return jsonify({'error': str(e)}), 500

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/ask', methods=['POST'])
def ask_question():
    global rag_model, gemini_model, history
    try:
        data = request.get_json()
        query = data.get('question', '').strip()
        
        if not query:
            return jsonify({'error': 'No question provided'}), 400
        
        if rag_model is None or gemini_model is None:
            logger.error("RAG pipeline is not initialized.")
            return jsonify({'error': 'RAG pipeline not initialized'}), 500
        
        relevant_chunks = retrieve_relevant_chunks(query, rag_model)
        answer = generate_answer(query, relevant_chunks, history, gemini_model)
        
        history.append({"question": query, "answer": answer})
        
        return jsonify({'answer': answer}), 200
    except Exception as e:
        logger.error(f"Error answering question: {e}")
        return jsonify({'error': str(e)}), 500


# if __name__ == '__main__':
if __name__ == '__main__':
    
    init_rag_pipeline()  # Initialize pipeline on startup
    # app.run(debug=True)
    app.run(host='0.0.0.0', port=5000)


