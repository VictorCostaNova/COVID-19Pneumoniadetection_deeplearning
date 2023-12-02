from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw
from io import BytesIO
from roboflow import Roboflow
from flask import Flask, render_template, request, send_from_directory

app = Flask(__name__)

# Configurações para uploads de arquivo
UPLOAD_FOLDER_PATH = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER_PATH
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Função para verificar extensões permitidas
def verifica_ext(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def identificando(image_path):
    rf = Roboflow(api_key="w5m2kG5xO7pQ50OTHcyk")
    project = rf.workspace().project("lungs-t9kpv")
    model = project.version(1).model

    # Executar a previsão no modelo
    prediction_group = model.predict(image_path, confidence=40, overlap=30)

    # Acessando a classe dentro da estrutura do PredictionGroup
    predicted_class = prediction_group.predictions[0].json_prediction['class']
    predicted_confidence = prediction_group.predictions[0].json_prediction['confidence']
    predicted_confidence_percent = round(predicted_confidence * 100, 2)
    return predicted_class, predicted_confidence_percent

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/descricao')
def descricao():
    return render_template('desc.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "Nenhuma imagem encontrada."

    file = request.files['image']

    if file.filename == '':
        return "Nenhum arquivo selecionado."

    if file and verifica_ext(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result,confianca = identificando(filepath)

        return render_template('index.html', result=result, confianca=confianca, image_url=filename)#, confianca=confianca

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER_PATH):
        os.mkdir(UPLOAD_FOLDER_PATH)
    app.run(debug=True, host='0.0.0.0')
