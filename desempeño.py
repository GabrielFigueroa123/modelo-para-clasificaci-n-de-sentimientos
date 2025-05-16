import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar el dataset
df = pd.read_csv("comentarios.csv")

X = df["Comentario"]
y = df["Sentimiento"]

# Vectorizacion del dataset
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.8, random_state=42)

# Entrenar un clasificador KNN
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

# Predecir en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular metricas
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average=None, labels=model.classes_)
precision = precision_score(y_test, y_pred, average=None, labels=model.classes_)
recall = recall_score(y_test, y_pred, average=None, labels=model.classes_)
matriz = confusion_matrix(y_test, y_pred, labels=model.classes_)

# Mostrar las metricas de desempeño
print(f"Exactitud: {accuracy:.4f}\n")

for i, label in enumerate(model.classes_):
    print(f"Clase: {label}")
    print(f"  Precisión: {precision[i]:.4f}")
    print(f"  Recall:    {recall[i]:.4f}")
    print(f"  F1:  {f1[i]:.4f}")
    print()

# Mostrar matriz de confusion
plt.figure(figsize=(8, 6))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
            xticklabels=model.classes_,
            yticklabels=model.classes_)
plt.xlabel("Real")
plt.ylabel("Prediccion")
plt.title("Matriz de Confusion")
plt.tight_layout()
plt.show()

