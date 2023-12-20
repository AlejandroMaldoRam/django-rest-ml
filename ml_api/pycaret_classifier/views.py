from django.shortcuts import render
from .apps import PycaretClassifierConfig
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import pandas as pd
from pycaret.classification import predict_model

# Create your views here.
class call_model(APIView):
    
    def post(self,request):
        print("Request post: ", request.data)
        if request.method == 'POST':
            df = pd.DataFrame.from_dict([request.data])
            print("DF: \n", df)

            predictions = predict_model(PycaretClassifierConfig.model, data=df)
            print("Predictions:\n", predictions)
            
            return JsonResponse({"prediction": predictions["prediction_label"].iloc[0], "score":predictions["prediction_score"].iloc[0]})

