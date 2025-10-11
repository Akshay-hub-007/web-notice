from django.shortcuts import render
from django.contrib.auth.decorators import login_required
# Create your views here.

@login_required
def dashboard(request):
    user=request.user

    return render(request,'dashboard.html',{'user':user})

@login_required
def notices(request):
    return render(request,'notices.html')

@login_required
def manage(request):
    return render(request,'manage.html')


def home(request):
    return render(request, "home.html")

@login_required
def creategroup(request):
    return render(request,'create.html')