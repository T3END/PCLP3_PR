import pandas as pd
import numpy as np
import random
import requests
import matplotlib.pyplot as plt
import seaborn as sns


def obtine_modele_marca(marca):
    url = f"https://vpic.nhtsa.dot.gov/api/vehicles/GetModelsForMake/{marca}?format=json"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        modele = [m['Model_Name'] for m in data['Results']]
        return modele[:5] if modele else ['Generic']
    else:
        return ['Generic']

marci = ['BMW', 'Ford', 'Toyota', 'Hyundai', 'Audi', 'Mercedes-Benz', 'Volkswagen']

modele_api = {}

for marca in marci:
    modele_api[marca] = obtine_modele_marca(marca)

for brand, modele in modele_api.items():
    print(f"{brand}: {modele}")