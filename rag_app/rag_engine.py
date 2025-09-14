import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import PyPDF2
from docx import Document as DocxDocument
from openai import OpenAI
from django.conf import settings

class RAGEngine:
    def __init__(self):
        # Load sentence transformer model for embeddings
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.text_chunks = []
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)
    
    def extract_text_from_pdf(self, file_path):
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def extract_text_from_docx(self, file_path):
        """Extract text from DOCX file"""
        text = ""
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text
    
    def extract_text_from_txt(self, file_path):
        """Extract text from TXT file"""
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
        return text
    
    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Split text into chunks for better processing"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        
        return chunks
    
    def process_document(self, file_path):
    
        # Determine file type and extract text
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            text = self.extract_text_from_pdf(file_path)
        elif file_extension == '.docx':
            text = self.extract_text_from_docx(file_path)
        elif file_extension == '.txt':
            text = self.extract_text_from_txt(file_path)
        else:
            return False
        
        if not text.strip():
            return False
        
        # Split text into chunks
        chunks = self.chunk_text(text)
        
        # Create embeddings for chunks
        embeddings = self.embedder.encode(chunks)
        
        # Add to FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
        
        self.index.add(np.array(embeddings).astype('float32'))
        self.text_chunks.extend(chunks)
        
        return True
    
    def search_similar_chunks(self, question, k=3):
        """Find most similar text chunks to the question"""
        if self.index is None or len(self.text_chunks) == 0:
            return []
        
        # Create embedding for question
        question_embedding = self.embedder.encode([question])
        
        # Search in FAISS index
        distances, indices = self.index.search(
            np.array(question_embedding).astype('float32'), k
        )
        
        # Return relevant chunks
        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                relevant_chunks.append(self.text_chunks[idx])
        
        return relevant_chunks
    
    def generate_answer(self, question, context_chunks):
        """Generate answer using OpenAI with retrieved context"""
        if not context_chunks:
            return "I don't have enough information to answer your question. Please upload relevant documents first."
        
        # Combine context chunks
        context = "\n\n".join(context_chunks)
        
        # Create prompt for OpenAI
        prompt = f"""Based on the following context, please answer the question accurately and concisely.

Context:
{context}

Question: {question}

Answer:"""
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on the provided context. If the context doesn't contain relevant information, say so."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=500,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Sorry, I encountered an error generating the answer: {str(e)}"
    
    def ask_question(self, question):
        """Main method to ask a question and get an answer"""
        # Find relevant chunks
        relevant_chunks = self.search_similar_chunks(question)
        
        # Generate answer using OpenAI
        answer = self.generate_answer(question, relevant_chunks)
        
        return answer

# Create global RAG engine instance
rag_engine = RAGEngine()