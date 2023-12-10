document.getElementById('predictForm').onsubmit = function(e) {
    e.preventDefault();
    
    // Coleta os valores dos inputs e criar um objeto
    let inputData = {
        'fixed_acidity': parseFloat(document.getElementById('fixed_acidity').value),
        'volatile_acidity': parseFloat(document.getElementById('volatile_acidity').value),
        'citric_acid': parseFloat(document.getElementById('citric_acid').value),
        'residual_sugar': parseFloat(document.getElementById('residual_sugar').value),
        'chlorides': parseFloat(document.getElementById('chlorides').value),
        'free_sulfur_dioxide': parseFloat(document.getElementById('free_sulfur_dioxide').value),
        'total_sulfur_dioxide': parseFloat(document.getElementById('total_sulfur_dioxide').value),
        'density': parseFloat(document.getElementById('density').value),
        'pH': parseFloat(document.getElementById('pH').value),
        'sulphates': parseFloat(document.getElementById('sulphates').value),
        'alcohol': parseFloat(document.getElementById('alcohol').value)
    };

    // Exibe resultados como um placeholder enquanto espera a resposta
    var resultContainer = document.getElementById('predictionResult');
    resultContainer.innerHTML = "Calculando...";
    
    // Faz o POST request para a API Flask com os valores obtidos
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(inputData),
    })
    .then(response => response.json())
    .then(data => {
        // Exibe a qualidade prevista
        resultContainer.innerHTML = 'Qualidade Prevista: ' + data.prediction;

        // Formata e exibe a probabilidade de cada classe
        var probaContainer = document.getElementById('predictionProba');
        var probaText = 'Probabilidade de cada classe:<br>';
        data.prediction_proba.forEach(function(p, index) {
            // As classes de qualidade variam de 3 a 8, então ajustamos os índices com +3
            var qualityClass = index + 3;
            probaText += `Classe ${qualityClass}: ${(p * 100).toFixed(2)}%<br>`;
        });
        probaContainer.innerHTML = probaText;
    })
    .catch(error => {
        console.error('Error:', error);
        resultContainer.innerHTML = "Error: " + error;
    });
};