import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setări vizuale
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Citim datele
df = pd.read_csv("train.csv")

# Analiza valorilor lipsa
print(" Valori lipsa:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Statistici descriptive
print("\n Statistici descriptive (numerice):")
print(df.describe())

print("\n Statistici descriptive (categorice):")
print(df.describe(include=['object']))

# Analiza distributiei variabilelor
coloane_num = ['an_model', 'kilometraj', 'capacitate_cilindirca', 'nr_usi', 'pret']
df[coloane_num].hist(bins=20, figsize=(14, 10))
plt.suptitle("Distributii pentru variabile numerice", fontsize=16)
plt.tight_layout()
plt.show()

coloane_cat = ['marca', 'model', 'combustibil', 'transmisie']
for z in coloane_cat:
    plt.figure()
    sns.countplot(data=df, x=z, order=df[z].value_counts().index)
    plt.xticks(rotation=45)
    plt.title(f"Distributia valorilor pentru {z}")
    plt.show()

# Detectarea outlierilor
for col in ['kilometraj', 'capacitate_cilindirca', 'pret']:
    plt.figure()
    sns.boxplot(data=df, x=col)
    plt.title(f"Boxplot pentru {col}")
    plt.show()

# Analiza corelatiilor
plt.figure()
corr = df[coloane_num].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Heatmap corelatii între variabile numerice")
plt.show()

# Analiza relatiilor cu variabila tintă
for col in ['an_model', 'kilometraj', 'capacitate_cilindirca']:
    plt.figure()
    sns.scatterplot(data=df, x=col, y='pret')
    plt.title(f"Relatia dintre {col} și pret")
    plt.show()