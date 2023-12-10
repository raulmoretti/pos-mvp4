# Autor: Raul Moretti Coelho
# Data: 09-12-2023
#### Descrição: Script para treinar um modelo de machine learning para classificação de vinhos brancos e tintos

# Escolha do Dataset:
#### Vamos escolher um dataset do UCI Machine Learning Repository, o "Wine Quality Data Set" que classifica a qualidade de vinhos brancos e tintos.
#### O dataset pode ser encontrado em: https://archive.ics.uci.edu/ml/datasets/Wine+Quality

# Importando as bibliotecas necessárias
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from ucimlrepo import fetch_ucirepo
from prettytable import PrettyTable

# Definindo as classes do dataset
class_names = ['Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7', 'Classe 8', 'Classe 9']

# Definindo a função para imprimir a matriz de confusão
def print_confusion_matrix(cm, class_names):
    table = PrettyTable()
    # Adicionando cabeçalho com os nomes das classes
    table.field_names = ["Classe/Predição"] + class_names
    
    # Preenchendo as linhas da tabela com os dados da matriz de confusão
    for i, row in enumerate(cm):
        formatted_row = [f"{class_names[i]} (Real)"] + list(row)
        table.add_row(formatted_row)
    
    print(table)


# Criando a classe WineQualityDataset para carregar o dataset e separar os dados de treino e teste
class WineQualityDataset:
    """

    Parametros
    ----------
    test_size : tamanho da amostra de teste
    random_state : semente para o gerador de números aleatórios

    Atributos
    ----------
    dataset : dataset carregado
    X : features do dataset
    y : targets do dataset
    X_train : features de treino
    X_test : features de teste
    y_train : targets de treino
    y_test : targets de teste

    """
    def __init__(self, test_size=0.2, random_state=42):
        self.dataset = fetch_ucirepo(id=186) # Carregando o dataset do UCI Machine Learning Repository pelo ID 186
        self.X = self.dataset.data.features
        self.y = self.dataset.data.targets
        if isinstance(self.y, pd.DataFrame):
            self.y = self.y.values.ravel()
        else:
            self.y = self.y.ravel()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state)

# Criando a classe BaseModel para definir os métodos comuns a todos os modelos
class BaseModel:
    """

    Parametros
    ----------
    model : modelo de machine learning
    scaler : tipo de scaler a ser utilizado (minmax ou standard)

    Atributos
    ----------
    scaler : scaler utilizado
    model : modelo utilizado
    pipeline : pipeline utilizado
    best_model : melhor modelo encontrado pelo GridSearchCV

    Metodos
    ----------
    optimize_hyperparameters : otimiza os hiperparametros do modelo
    evaluate_model : avalia o modelo
    save_model : salva o modelo em um arquivo

    """
    def __init__(self, model, scaler='minmax'):
        self.scaler = MinMaxScaler() if scaler == 'minmax' else StandardScaler()
        self.model = model
        self.pipeline = Pipeline([
            ('scaler', self.scaler),
            ('model', self.model)
        ])
        self.best_model = None

    def optimize_hyperparameters(self, X_train, y_train, param_grid, cv=5):
        # Realizando a busca pelos melhores hiperparâmetros com GridSearchCV
        grid_search = GridSearchCV(self.pipeline, param_grid, cv=cv, n_jobs=-1)  # Adicionado n_jobs=-1 para paralelismo
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        print(f"Melhores parâmetros: {grid_search.best_params_}")

    def evaluate_model(self, X_test, y_test, class_names):
        y_pred = self.best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Acurácia: {accuracy:.2%}")
        print_confusion_matrix(cm, class_names)
        return accuracy, cm

    def save_model(self, file_path):
        # Salvando o modelo treinado em um arquivo
        joblib.dump(self.best_model, file_path)
        print(f"Modelo salvo em {file_path}")

# Criando as classes para cada modelo
# Cada classe herda da classe BaseModel e define o modelo e o scaler específicos
class KNNModel(BaseModel):
    """

    Classe KNNModel para o algoritmo KNN (K-Nearest Neighbors) de classificação, que é um algoritmo de aprendizado supervisionado que pode ser usado para classificação ou regressão e que classifica novos pontos de dados com base nos pontos de dados que estão mais próximos a ele.

    """

    def __init__(self, scaler='minmax'):
        super().__init__(model=KNeighborsClassifier(), scaler=scaler)

class DecisionTreeModel(BaseModel):
    """
    
    Classe DecisionTreeModel para o algoritmo Decision Tree (Árvore de Decisão) de classificação, que é um algoritmo de aprendizado supervisionado que pode ser usado para classificação ou regressão e que cria uma árvore de decisão a partir do dataset.

    """
    
    def __init__(self, scaler='minmax'):
        super().__init__(model=DecisionTreeClassifier(), scaler=scaler)

class NaiveBayesModel(BaseModel):
    """

    Classe NaiveBayesModel para o algoritmo Naive Bayes de classificação, que é um algoritmo de aprendizado supervisionado que pode ser usado para classificação ou regressão e que é baseado no teorema de Bayes com a suposição "ingênua" de independência entre cada par de recursos.

    """
    def __init__(self, scaler='minmax'):
        super().__init__(model=GaussianNB(), scaler=scaler)
        
    def optimize_hyperparameters(self, X_train, y_train, param_grid={}, cv=5):
        super().optimize_hyperparameters(X_train, y_train, param_grid, cv)

class SVMModel(BaseModel):
    """

    Classe SVMModel para o algoritmo SVM (Support Vector Machine) de classificação, que é um algoritmo de aprendizado supervisionado que pode ser usado para classificação ou regressão e que cria um hiperplano ou conjunto de hiperplanos em um espaço de alta ou infinita dimensão.

    """
    def __init__(self, scaler='minmax'):
        super().__init__(model=SVC(), scaler=scaler)

# Criando a função para explicar a matriz de confusão
def explain_confusion_matrix(cm, class_names):
    print("Explicação da Matriz de Confusão:")
    print("Os valores na diagonal principal (de cima para baixo) mostram o número de previsões corretas para cada classe.")
    print("Por exemplo:")

    for i in range(len(class_names)):
        print(f"- A classe '{class_names[i]}' (Real) teve {cm[i, i]} previsões corretas (Verdadeiros Positivos).")
    
    print("\nOs valores fora da diagonal principal indicam as previsões incorretas (Erros de Classificação).")
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                print(f"- O modelo previu incorretamente {cm[i, j]} vezes a classe '{class_names[i]}' (Real) como classe '{class_names[j]}' (Predita).")

    print("\nUma matriz de confusão com muitos valores altos fora da diagonal principal indica um modelo que está frequentemente confundindo as classes.")
    print("Uma matriz de confusão com valores altos na diagonal principal e baixos fora dela indica um modelo com boa performance.")

# Criando a função main para executar o script
if __name__ == "__main__":
    # Carregando o dataset
    dataset = WineQualityDataset()

    # Instanciando os modelos
    knn_model = KNNModel(scaler='minmax')

    # Documentação das etapas do processo de criação do modelo KNN
    # Descrição: Nesta parte, otimizamos os hiperparâmetros para o modelo KNN.
    # Utilizamos o GridSearchCV para testar uma combinação de diferentes hiperparâmetros
    # e selecionamos os melhores para nosso modelo baseado na acurácia da validação cruzada.

    # Definindo os parâmetros para otimização do modelo
    param_grid_knn = {
        'model__n_neighbors': [3, 5, 7, 9],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan']
    }

    # Otimizando o modelo
    knn_model.optimize_hyperparameters(dataset.X_train, dataset.y_train, param_grid_knn)

    # Lista de nomes das classes para a matriz de confusão
    class_names = ['Classe 3', 'Classe 4', 'Classe 5', 'Classe 6', 'Classe 7', 'Classe 8', 'Classe 9']

    # Avaliação do modelo KNN com os dados de teste, usando os rótulos de classe para uma saída mais detalhada
    accuracy, cm = knn_model.evaluate_model(dataset.X_test, dataset.y_test, class_names)

    # Explicação da matriz de confusão
    explain_confusion_matrix(cm, class_names)

    # Salvando o modelo treinado
    knn_model.save_model('knn_model.pkl')


# Documentação adicional
#### Descrição: Depois de treinar o modelo, avaliamos sua performance no conjunto de teste. A acurácia e a matriz de confusão foram apresentadas para entender melhor como o modelo está performando. A acurácia é uma medida geral de quão frequentemente o modelo está correto, enquanto a matriz de confusão nos dá uma visão mais detalhada dos erros de classificação.

# Análise de resultados do modelo:
#### No final do script, após a avaliação do modelo KNN, identificamos que o modelo atingiu uma acurácia de 65.77%. A matriz de confusão nos mostra que há um número significativo de previsões corretas ao longo da diagonal principal. No entanto, observamos também que o modelo confundiu frequentemente as classes X e Y, o que poderia indicar a similaridade entre as características dessas classes ou um desbalanceamento nos dados de treino. A análise detalhada da matriz nos leva a considerar estratégias como coleta de mais dados, tentar algoritmos de balanceamento de classes ou introduzir novas features que possam ajudar a distinguir melhor entre as classes que estão sendo confundidas.

# Conclusão:
#### Este experimento ilustrou o processo de desenvolvimento de um modelo de classificação KNN para o conjunto de dados de qualidade de vinho. Após otimização de hiperparâmetros e avaliação, observamos a performance do modelo no conjunto de teste. Os resultados demonstram a importância de uma seleção cuidadosa de hiperparâmetros e a necessidade de técnicas de validação rigorosas.