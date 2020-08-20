from django.urls import path, include
from demoapp1 import views

app_name = 'demoapp1'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('about_me/', views.AboutMeView.as_view(), name='about_me'),
    path('plotter/', views.PlotterView.as_view(), name='plotter'),
    path('calculator/', views.calculator, name='calculator'),
]
