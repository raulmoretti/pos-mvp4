# Aplicação de Previsão de Qualidade de Vinho

Este projeto é uma aplicação web Flask que utiliza um modelo de machine learning treinado para prever a qualidade de amostras de vinho.

## Pré-requisitos

Antes de iniciar, você precisará do Python instalado em seu sistema. Este projeto foi desenvolvido com Python 3.10.12, mas deve funcionar com outras versões que suportam as mesmas bibliotecas.

Recomenda-se o uso de um ambiente virtual para instalar e executar o projeto.

## Configuração

Siga as instruções abaixo para configurar o projeto em sua máquina local:

1. Clone o repositório do GitHub onde o projeto está hospedado:

```bash
git clone https://github.com/raulmoretti/pos-mvp4
cd pos-mvp4
```

2. Crie um ambiente virtual e ative-o:

```bash
python -m venv venv
source venv/bin/activate  # No Windows use `venv\Scripts\activate`
```

3. Instale as dependências do projeto:

```bash
pip install -r requirements.txt
```

## Executando a aplicação

Após instalar as dependências, você está pronto para executar a aplicação:

```bash
python3 app.py
```

A aplicação estará disponível em `http://localhost:5000/` no seu navegador.

## Usando a aplicação

Para usar a aplicação web, siga estes passos:

1. Abra `http://localhost:5000/` no seu navegador.
2. Insira os parâmetros solicitados nos campos do formulário.
3. Clique no botão "Prever Qualidade" para obter a previsão.
4. A qualidade prevista do vinho será exibida na página.

## Executando testes

Para executar os testes automatizados e verificar a acurácia do modelo:

```bash
cd modelo-ml
pytest test_model_performance.py
