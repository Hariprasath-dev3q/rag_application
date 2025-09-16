import os
import faiss
import numpy as np
import PyPDF2
from docx import Document as DocxDocument
from openai import OpenAI
from django.conf import settings

class RAGEngine:
    def __init__(self):
        self.embedder = None   # it don’t load immediately
        self.index = None
        self.text_chunks = []
        self.openai_client = OpenAI(api_key=settings.OPENAI_API_KEY)

    def load_model(self):
        """Load sentence transformer only when needed"""
        if self.embedder is None:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_text_from_pdf(self, file_path):
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
        text = ""
        try:
            doc = DocxDocument(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        except Exception as e:
            print(f"Error reading DOCX: {e}")
        return text

    def extract_text_from_txt(self, file_path):
        text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except Exception as e:
            print(f"Error reading TXT: {e}")
        return text

    def chunk_text(self, text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    def process_document(self, file_path):
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

        chunks = self.chunk_text(text)

        # ✅ lazy load
        self.load_model()
        embeddings = self.embedder.encode(chunks)

        if self.index is None:
            self.index = faiss.IndexFlatL2(embeddings.shape[1])

        self.index.add(np.array(embeddings).astype('float32'))
        self.text_chunks.extend(chunks)

        return True

    def search_similar_chunks(self, question, k=3):
        if self.index is None or len(self.text_chunks) == 0:
            return []

        # ✅ lazy load
        self.load_model()
        question_embedding = self.embedder.encode([question])

        distances, indices = self.index.search(
            np.array(question_embedding).astype('float32'), k
        )

        relevant_chunks = []
        for idx in indices[0]:
            if idx < len(self.text_chunks):
                relevant_chunks.append(self.text_chunks[idx])

        return relevant_chunks

    def generate_answer(self, question, context_chunks):
        if not context_chunks:
            return "I don't have enough information to answer your question. Please upload relevant documents first."

        context = "\n\n".join(context_chunks)
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
        relevant_chunks = self.search_similar_chunks(question)
        return self.generate_answer(question, relevant_chunks)


# ✅ Create global RAG engine instance
rag_engine = RAGEngine()
