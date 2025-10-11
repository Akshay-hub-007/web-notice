from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),         # 👈 root URL points to home page
    path('dashboard/', views.dashboard, name="dashboard"),
    path('notices/', views.notices, name='notices'),
    path('manage/', views.manage, name='manage'),
    path('creategroup/',views.creategroup,name='creategroup')
]
