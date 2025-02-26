import os
import numpy as np
from PIL import Image
import json
from flask import Flask, request, jsonify, render_template

#Luo tarvittavat tiedostot/varmistaa niiden olemassaolon
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
VECTOR_FILE = 'vectors.json'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#vektoroi
#silleen että ottaa vaan tietyt pikselit kuvasta --> katso netistä miten saa keskitettyä siihen
#poista tausta kuvista jos mahdollista
#katso yolov8 custom database 
def image_to_vector(image_path):
    with Image.open(image_path) as img:
        img = img.resize((32, 32))
        vector = np.array(img).flatten().tolist()
    return vector

def save_vector_to_file(file_name, vector):
    data = {}
    if os.path.exists(VECTOR_FILE):
        with open(VECTOR_FILE, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = {}
    
    data[file_name] = ','.join(map(str, vector))  # Tallennetaan vektorit merkkijonona yhteen riville
    
    with open(VECTOR_FILE, 'w') as f:
        json.dump(data, f, indent=4)

#käyttää html sivua
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    
    # Varmistetaan, että tiedosto tallennetaan eri nimellä, jos samanniminen on jo olemassa
    base, ext = os.path.splitext(file.filename)
    counter = 1
    while os.path.exists(file_path):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{base}_{counter}{ext}")
        counter += 1
    
    file.save(file_path)
    
    vector = image_to_vector(file_path)
    save_vector_to_file(os.path.basename(file_path), vector)
    
    return jsonify({"vector": vector, "file_path": file_path})

if __name__ == '__main__':
    app.run(debug=True)
