from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_document, name='upload_document'),
    path('ask/', views.ask_question, name='ask_question'),
    path('delete/<int:document_id>/', views.delete_document, name='delete_document'),
]