import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

X_train = df_train.drop('pret', axis=1)
y_train = df_train['pret']
X_test = df_test.drop('pret', axis=1)
y_test = df_test['pret']

# Transformam regresia intr-un model de clasificare
threshold = y_train.median()
y_train_class = (y_train >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

categorical_cols = ['marca', 'model', 'combustibil', 'transmisie']
numerical_cols = ['an_model', 'kilometraj', 'capacitate_cilindrica', 'nr_usi']

# Preprocesare - OneHotEncoder
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'
)

# Pipeline cu LinearRegression
model_regression = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Antrenare model de regresie
model_regression.fit(X_train, y_train)

# Predictii regresie
y_pred = model_regression.predict(X_test)

# Evaluare model regresie
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Evaluare model:")
print(f"RMSE: {rmse:.2f} EUR")
print(f"MAE:  {mae:.2f} EUR")
print(f"R²:   {r2:.3f}")

# Evaluare clasificare
y_pred_class = (y_pred >= threshold).astype(int)

# Calcul metrici clasificare
accuracy = accuracy_score(y_test_class, y_pred_class)
precision = precision_score(y_test_class, y_pred_class)
recall = recall_score(y_test_class, y_pred_class)
f1 = f1_score(y_test_class, y_pred_class)

print("\nEvaluare model Clasificare:")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-score:  {f1:.2f}")

# Real vs Prezis (Regresie)
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Pret real")
plt.ylabel("Pret prezis")
plt.title("Comparare: Pret real vs. Pret prezis (Regresie)")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')
plt.grid(True)
plt.tight_layout()
plt.show()

# Distributia erorilor (Reziduuri)
erori = y_test - y_pred
plt.figure(figsize=(8,5))
sns.histplot(erori, kde=True, bins=30)
plt.title("Distributia erorilor (Reziduuri)")
plt.xlabel("Eroare (pret real - prezis)")
plt.tight_layout()
plt.show()

# Matrice de confuzie pentru clasificare
fig, ax = plt.subplots(figsize=(6,6))
ConfusionMatrixDisplay.from_predictions(y_test_class, y_pred_class, cmap="Blues", ax=ax)
plt.title("Matrice de confuzie - Clasificare binara")
plt.show()