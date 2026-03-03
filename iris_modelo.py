from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

print("🌸 Modelo de Clasificación Iris\n")

# 1️⃣ Cargar dataset
iris = load_iris()
X = iris.data      # Características
y = iris.target    # Etiquetas

# 2️⃣ Dividir datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3️⃣ Crear modelo
modelo = KNeighborsClassifier(n_neighbors=3)

# 4️⃣ Entrenar modelo
modelo.fit(X_train, y_train)

# 5️⃣ Predecir
predicciones = modelo.predict(X_test)

# 6️⃣ Evaluar precisión
accuracy = accuracy_score(y_test, predicciones)

print("Precisión del modelo:", round(accuracy * 100, 2), "%")