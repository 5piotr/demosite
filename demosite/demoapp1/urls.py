from django.urls import path, include
from demoapp1 import views

app_name = 'demoapp1'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('about_me/', views.AboutMeView.as_view(), name='about_me'),
    path('calculator/', views.calculator, name='calculator'),
    path('graphing_calculator/', views.graphing_calculator, name='graphing_calculator'),
    path('under_construction/',views.under_construction, name='under_construction'),
]
