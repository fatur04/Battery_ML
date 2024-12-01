from django.shortcuts import render
from .forms import BatteryPredictionForm
from .ml_models import load_and_train_models, rf_model_class, rf_model_reg, lstm_model, scaler, evaluate_models

import numpy as np

def estimate_battery_percentage(volt):
    if volt >= 14.0:          # Fully charged or above
        return 100
    elif volt >= 13.5:        # High charge
        return 90
    elif volt >= 13.0:        # Nearly full
        return 80
    elif volt >= 12.6:        # Good charge
        return 70
    elif volt >= 12.4:        # Moderate charge
        return 60
    elif volt >= 12.0:        # Low charge
        return 50
    elif volt >= 11.8:        # Very low charge
        return 40
    elif volt >= 11.5:        # Critical level
        return 25
    elif volt >= 11.0:        # Dangerously low
        return 10
    else:                      # Below critical level
        return 5    
    
def home(request):
    prediction = ""
    accuracy_data = None
    form = BatteryPredictionForm()

    if request.method == 'POST' and 'predict' in request.POST:
        # Prediction logic
        form = BatteryPredictionForm(request.POST)
        if form.is_valid():
            volt = form.cleaned_data['volt']
            arus = form.cleaned_data['arus']
            #baterai = form.cleaned_data['baterai']
            suhu = form.cleaned_data['suhu']
            kelembaban = form.cleaned_data['kelembaban']

            # Estimate battery percentage based on voltage
            battery_percentage = estimate_battery_percentage(volt)

            new_data = scaler.transform([[volt, arus, battery_percentage, suhu, kelembaban, 0]])
            condition_pred = rf_model_class.predict(new_data)
            condition = "Good" if condition_pred == 1 else "Bad"

            lifetime_pred = rf_model_reg.predict(new_data)[0]
            if condition == "Bad" or volt < 11.9:
                lifetime_pred = 0

            hours_duration = (65 / arus) * (battery_percentage / 100) if condition == "Good" else 0
            prediction = f"Battery Condition: {condition}, Battery Lifetime: {lifetime_pred:.2f} Years, Duration: {hours_duration:.2f} Hours"

    elif 'train' in request.POST:
        # Training logic and evaluate models
        load_and_train_models()
        accuracy_data = evaluate_models()

    return render(request, 'home.html', {'form': form, 'prediction': prediction, 'accuracy_data': accuracy_data})


def train_model(request):
    X_test_class, y_test_class, X_test_reg, y_test_reg = load_and_train_models()
    accuracy_data = evaluate_models(X_test_class, y_test_class, X_test_reg, y_test_reg)
    return render(request, 'train.html', {'message': "Model telah dilatih ulang!", 'accuracy_data': accuracy_data})


