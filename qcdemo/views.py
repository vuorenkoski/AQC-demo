from django.shortcuts import render
from qcdemo.utils import algorithms

def index(request):
    return render(request, 'index.html', {'algorithms': algorithms})

def docs(request):
    return render(request, 'docs.html', {'algorithms': algorithms}) 
