from django.urls import path
from .views import home, train_model

urlpatterns = [
    path('', home, name='home'),
    path('train/', train_model, name='train_model'),
]
