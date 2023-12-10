import joblib
import pytest
from sklearn.metrics import accuracy_score
from modelo import WineQualityDataset

# Definindo uma variável global para o limiar de acurácia mínimo aceitável
MIN_ACCURACY_THRESHOLD = 0.65

# Carrega o conjunto de dados de teste e os rótulos verdadeiros
@pytest.fixture(scope="session")
def test_data():
    dataset = WineQualityDataset()
    return dataset.X_test, dataset.y_test

# Teste para assegurar que o modelo atende aos requisitos de desempenho estabelecidos
def test_model_accuracy(test_data):
    model = joblib.load('knn_model.pkl')
    X_test, y_test = test_data
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    assert accuracy >= MIN_ACCURACY_THRESHOLD, f"Desempenho do modelo abaixo do limiar mínimo de acurácia de {MIN_ACCURACY_THRESHOLD*100}%"

# Teste adicional para verificar se as probabilidades máximas correspondem às classes previstas
def test_model_probabilities(test_data):
    model = joblib.load('knn_model.pkl')
    X_test, _ = test_data
    y_pred = model.predict(X_test)
    proba_pred = model.predict_proba(X_test)
    
    # Obtem as classes com a maior probabilidade de cada previsão
    highest_proba_indices = proba_pred.argmax(axis=1)
    # Verifica se as classes preditas correspondem àquelas com a maior probabilidade
    for i, predicted_class in enumerate(y_pred):
        class_label_from_proba = highest_proba_indices[i] + 3  # Somamos 3 para ajustar ao intervalo correto das classes que vai de 3 a 8
        assert predicted_class == class_label_from_proba, (
            f"Classe predita {predicted_class}, probabilidade máxima pertence à classe {class_label_from_proba}"
        )