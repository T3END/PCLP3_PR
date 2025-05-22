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

random.seed(42)
np.random.seed(42)
modele_api = {marca: obtine_modele_marca(marca) for marca in marci}
combustibil = ['benzina', 'diesel', 'hybrid']
transmisie = ['manuala', 'automata']
ani_disponibili = list(range(2005, 2024))

    #GENERARE MASINI
def genereaza_masini(n=700):
    data = []
    for _ in range(n):
        marca = random.choice(marci)
        model = random.choice(modele_api[marca])
        an_model = random.choice(ani_disponibili)
        kilometraj = np.random.randint(5000, 300000)
        capacitate_cilindirca = round(np.random.uniform(1.0, 4.5), 1)
        combustibil_val = random.choice(combustibil)
        transmisie_val = random.choice(transmisie)
        nr_usi = random.choice([2, 3, 4, 5])

        pret_de_baza = 20000
        depreciere = (2024 - an_model) * 700
        impact_kilometraj = kilometraj * 0.03
        bonus_motor = capacitate_cilindirca * 1000
        bonus_marca = {
            'BMW': 3000, 'Mercedes-Benz': 3000, 'Audi': 2500,
            'Toyota': 1500, 'Volkswagen': 1500,
            'Ford': 1000, 'Hyundai': 500
        }.get(marca, 0)

        pret = pret_de_baza - depreciere - impact_kilometraj + bonus_motor + bonus_marca
        pret = max(1500, round(pret + np.random.normal(0, 1000)))

        data.append([marca, model, an_model, kilometraj, capacitate_cilindirca, combustibil_val, transmisie_val, nr_usi, pret])

    return pd.DataFrame(data, columns=[
        'marca', 'model', 'an_model', 'kilometraj', 'capacitate_cilindirca',
        'combustibil', 'transmisie', 'nr_usi', 'pret'
    ])

df = genereaza_masini(700)
df_train = df.sample(500, random_state=42)
df_test = df.drop(df_train.index)
df_train.to_csv("train.csv", index=False)
df_test.to_csv("test.csv", index=False)