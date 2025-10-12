from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),         # 👈 root URL points to home page
    path('dashboard/', views.dashboard, name="dashboard"),
    path('view_notices/', views.notices, name='view_notices'),
    path('create_notice/', views.create, name='create_notice'),
    # path('creategroup/',views.creategroup,name='creategroup')
]
