from flask import Flask, request, jsonify, render_template
from roboflow import Roboflow
import os
from dotenv import load_dotenv

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')

# Roboflow API
rf = Roboflow(api_key=api_key)
project = rf.workspace().project("mushroom-xtwbh")
model = project.version(1).model

#toisen mallin käyttöönotto odottaa vielä lol
#rf2 = Roboflow(api_key=API_KEY)
#project2 = rf2.workspace().project("fungeye")
#model2 = project2.version(2).model 

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

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

    prediction = model.predict(file_path).json()

    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)
