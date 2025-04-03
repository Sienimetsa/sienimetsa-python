from flask import Flask, request, jsonify, render_template, send_file
from roboflow import Roboflow
import supervision as sv
import cv2
import numpy as np
import os
from dotenv import load_dotenv


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
ANNOTATED_FOLDER = 'annotated'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(ANNOTATED_FOLDER, exist_ok=True)


load_dotenv()
api_key = os.getenv('ROBOFLOW_API_KEY')


rf1 = Roboflow(api_key=api_key)
project1 = rf1.workspace().project("mushroom-xtwbh")
model1 = project1.version(1).model

rf2 = Roboflow(api_key=api_key)
project2 = rf2.workspace().project("fungeye")
model2 = project2.version(2).model

rf3 = Roboflow(api_key="DLcTg0ZnoZqc6Q4v2kiQ")
project3 = rf3.workspace().project("task2-hfjmv")
model3 = project3.version(2).model


rf4 = Roboflow(api_key="DLcTg0ZnoZqc6Q4v2kiQ")
project4 = rf4.workspace().project("mushrooms-d36vk")
model4 = project4.version(3).model

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

    
    prediction1 = model1.predict(file_path, confidence=30, overlap=30).json()
    prediction2 = model2.predict(file_path, confidence=30, overlap=30).json()
    prediction3 = model3.predict(file_path, confidence=40, overlap=30).json()
    prediction4 = model4.predict(file_path, confidence=40, overlap=30).json()

    
    def extract_detections(prediction):
        if not prediction["predictions"]:  
            return sv.Detections.empty(), []  
        
        boxes = np.array([
            [p["x"] - p["width"] / 2, p["y"] - p["height"] / 2, 
             p["x"] + p["width"] / 2, p["y"] + p["height"] / 2]
            for p in prediction["predictions"]
        ])

        labels = [p["class"] for p in prediction["predictions"]]
        return sv.Detections(xyxy=boxes, class_id=np.zeros(len(boxes), dtype=int)), labels

    detections1, labels1 = extract_detections(prediction1)
    detections2, labels2 = extract_detections(prediction2)
    detections3, labels3 = extract_detections(prediction3)
    detections4, labels4 = extract_detections(prediction4)

    
    image = cv2.imread(file_path)

    
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

   
    if not detections1.is_empty():
        image = box_annotator.annotate(scene=image, detections=detections1)
        image = label_annotator.annotate(scene=image, detections=detections1, labels=labels1)

    if not detections2.is_empty():
        image = box_annotator.annotate(scene=image, detections=detections2)
        image = label_annotator.annotate(scene=image, detections=detections2, labels=labels2)
    
    if not detections3.is_empty():
        image = box_annotator.annotate(scene=image, detections=detections3)
        image = label_annotator.annotate(scene=image, detections=detections3, labels=labels3)

    if not detections4.is_empty():
        image = box_annotator.annotate(scene=image, detections=detections4)
        image = label_annotator.annotate(scene=image, detections=detections4, labels=labels4)

    
    annotated_path = os.path.join(ANNOTATED_FOLDER, "annotated_" + file.filename)
    cv2.imwrite(annotated_path, image)

    return jsonify({
        "model1_prediction": prediction1,
        "model2_prediction": prediction2,
        "model3_prediction": prediction3,
        "model4_prediction": prediction4,
        "annotated_image_url": f"/annotated/{file.filename}"
    })

@app.route('/annotated/<filename>')
def get_annotated_image(filename):
    return send_file(os.path.join(ANNOTATED_FOLDER, "annotated_" + filename), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)