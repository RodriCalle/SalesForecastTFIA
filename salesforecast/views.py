import csv
from datetime import datetime
import os
from unittest import result
from django.shortcuts import render
from django.views import View
import numpy as np
from salesforecast.TF.predict import predict

from salesforecast.forms import UploadForm

# Create your views here.

class SalesForecast(View):
    def get(self, request):
        form = UploadForm()
        return render(request, 'salesforecast/index.html', {'form': form, 'result': False})

    def post(self, request):
        arreglito = []

        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            original_file_path = 'static/files/' + request.FILES['file'].name
            updated_file_path = 'static/files/' + 'data_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'

            os.rename(original_file_path, updated_file_path)

            predict(updated_file_path)

            file = open('static/files/pronostico.csv', 'r')
            reader = csv.reader(file, delimiter=',')
            for i, row in enumerate(reader):
                if i != 0:
                    data = Data(row[0], row[1])
                    arreglito.append(data)

            return render(request, 'salesforecast/index.html', {
            'result': True, 
            'img_path_1': 'images/validate.png',
            'img_path_2': 'images/validateloss.png',
            'arreglito': arreglito
            })
        else:
            return render(request, 'salesforecast/index.html', {
                'form': form,
                'result': False
                })

class Data():
    def __init__(self, dato1, dato2):
        self.dato1 = dato1
        self.dato2 = dato2
