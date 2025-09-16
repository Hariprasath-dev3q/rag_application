from django.shortcuts import render, redirect
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import json
from .models import Document, ChatHistory
from .rag_engine import rag_engine
#import tempfile

# Create your views here.

def home(request):
    """Home page view"""
    documents = Document.objects.all().order_by('-uploaded_at')
    chat_history = ChatHistory.objects.all()[:5]  # Last 5 conversations
    
    context = {
        'documents': documents,
        'chat_history': chat_history,
    }
    return render(request, 'rag_app/home.html', context)

def upload_document(request):
    """Handle document upload"""
    if request.method == 'POST':
        if 'document' not in request.FILES:
            messages.error(request, 'No file selected!')
            return redirect('home')
        
        uploaded_file = request.FILES['document']
        
        # Check file type
        allowed_extensions = ['.pdf', '.docx', '.txt']
        file_extension = uploaded_file.name.lower().split('.')[-1]
        
        if f'.{file_extension}' not in allowed_extensions:
            messages.error(request, 'Please upload only PDF, DOCX, or TXT files!')
            return redirect('home')

        # Save the file temporarily
        '''with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            for chunk in uploaded_file.chunks():
                tmp.write(chunk)
            tmp_path = tmp.name'''
        
        # Save document
        document = Document(
            title=uploaded_file.name,
            file=uploaded_file
        )
        document.save()
        
        # Process document with RAG engine
        try:
            success = rag_engine.process_document(document.file.path)
            if success:
                document.processed = True
                document.save()
                messages.success(request, f'Document "{uploaded_file.name}" uploaded and processed successfully!')
            else:
                messages.error(request, 'Failed to process the document. Please check the file format.')
        except Exception as e:
            messages.error(request, f'Error processing document: {str(e)}')
    
    return redirect('home')

@csrf_exempt
def ask_question(request):
    """Handle question asking via AJAX"""
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            question = data.get('question', '').strip()
            
            if not question:
                return JsonResponse({'error': 'Please enter a question!'})
            
            # Get answer from RAG engine
            answer = rag_engine.ask_question(question)
            
            # Save to chat history
            ChatHistory.objects.create(
                question=question,
                answer=answer
            )
            
            return JsonResponse({
                'answer': answer,
                'success': True
            })
        
        except Exception as e:
            return JsonResponse({'error': f'Error: {str(e)}'})
    
    return JsonResponse({'error': 'Invalid request method'})

def delete_document(request, document_id):
    """Delete a document"""
    try:
        document = Document.objects.get(id=document_id)
        document.delete()
        messages.success(request, f'Document "{document.title}" deleted successfully!')
    except Document.DoesNotExist:
        messages.error(request, 'Document not found!')
    
    return redirect('home')



