import pandas as pd
import os

def load_data():
    data = pd.read_csv('datasets/celsius_fahrenheit_data.csv')
    celsius = data['Celsius'].values
    fahrenheit = data['Fahrenheit'].values
    return celsius, fahrenheit