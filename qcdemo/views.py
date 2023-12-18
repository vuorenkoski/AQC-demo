from django.shortcuts import render

def index(request):
    return render(request, 'index.html') 

def docs(request):
    return render(request, 'docs.html') 
