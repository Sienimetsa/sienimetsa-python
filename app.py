import os
import numpy as np
from PIL import Image
import json
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#muuntaa kuvan vektoriksi
def image_to_vector(image_path):
    with Image.open(image_path) as img:
        img = img.convert("L") 
        img = img.resize((32, 32))
        vector = np.array(img).flatten().tolist()
    return vector

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    vector = image_to_vector(file_path)
    
    return jsonify({"vector": vector, "file_path": file_path})

if __name__ == '__main__':
    app.run(debug=True)
