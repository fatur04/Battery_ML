from django import forms

class BatteryPredictionForm(forms.Form):
    volt = forms.FloatField(label='Voltage (V)', required=True)
    arus = forms.FloatField(label='Current (A)', required=True)
    #baterai = forms.FloatField(label='Battery Percentage (0-100)', required=True)
    suhu = forms.FloatField(label='Temperature (°C)', required=True)
    kelembaban = forms.FloatField(label='Humidity (%)', required=True)
