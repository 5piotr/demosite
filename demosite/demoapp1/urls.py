from django.urls import path, include
from demoapp1 import views

app_name = 'demoapp1'

urlpatterns = [
    path('', views.IndexView.as_view(), name='index'),
    path('about/', views.AboutView.as_view(), name='about'),
    path('calculator/', views.calculator, name='calculator'),
    path('graphing_calculator/', views.graphing_calculator, name='graphing_calculator'),
    path('simple_gesture_recognition/', views.simple_gesture_recognition, name='simple_gesture_recognition'),
    path('rps_cnn_details/', views.rps_cnn_details, name='rps_cnn_details'),
    path('apartment_price_estimator/', views.apartment_price_estimator, name='apartment_price_estimator'),
    path('apartment_estimator_details/', views.apartment_estimator_details, name='apartment_estimator_details'),
    path('privacy_policy/', views.privacy_policy, name='privacy_policy'),
    path('under_construction/',views.under_construction, name='under_construction'),
]
