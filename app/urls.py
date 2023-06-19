
from django.urls import path, include
from . import views

from app import views
# http://127.0.0.1:8000/api/t1/
urlpatterns = [
    path('t1/', views.index, name='index'),
    path('year/', views.return_year, name='return_year'),
    path('cv/display', views.returnvideo, name='returnvideo'),
    path('fileUpload/upload/', views.returnfile, name='returnfile'),
    path('download/',views.download,name='download'),
    path('rundance/',views.runDance,name='runDance'),
]