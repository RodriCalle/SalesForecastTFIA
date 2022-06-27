from django.urls import path

from salesforecast.views import SalesForecast

urlpatterns = [
    path('', SalesForecast.as_view()),
]