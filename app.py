from flask import Flask, request, render_template, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Carrega o modelo de machine learning
model = joblib.load('modelo-ml/knn_model.pkl')

@app.route('/')
def index():
    # Renderiza o front-end
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtem os dados do POST request
    data = request.get_json()
    
    # Transforma os dados em DataFrame
    df = pd.DataFrame(data, index=[0])
    
    # Faz previsão com o modelo
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
    
    # Retorna a resposta
    response = {
        'prediction': int(prediction[0]),
        'prediction_proba': prediction_proba.tolist()[0]
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)