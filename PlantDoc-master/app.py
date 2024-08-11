import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms.functional as TF
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

# Assuming CNN is a custom module you've created
import CNN

# Load CSV files
disease_info = pd.read_csv('disease_info.csv', encoding='cp1252')
supplement_info = pd.read_csv('supplement_info.csv', encoding='cp1252')

# Initialize the model
model = CNN.CNN(39)

# Load the model
try:
    state_dict = torch.load("plant_disease_model_1_latest.pt", map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading the model: {e}")
    # Optionally, exit the program if the model is crucial
    # import sys
    # sys.exit(1)

def prediction(image_path):
    try:
        image = Image.open(image_path)
        image = image.resize((224, 224))
        input_data = TF.to_tensor(image)
        input_data = input_data.view((-1, 3, 224, 224))
        output = model(input_data)
        output = output.detach().numpy()
        index = np.argmax(output)
        return index
    except Exception as e:
        print(f"Error in prediction: {e}")
        return None

app = Flask(__name__)
CORS(app)

@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/contact')
def contact():
    return render_template('contact-us.html')

@app.route('/index')
def ai_engine_page():
    return render_template('index.html')

@app.route('/mobile-device')
def mobile_device_detected_page():
    return render_template('mobile-device.html')

@app.route('/submit', methods=['GET', 'POST'])
def submit():
    if request.method == 'POST':
        try:
            image = request.files['image']
            filename = image.filename
            upload_folder = 'static/uploads'
            file_path = os.path.join(upload_folder, filename)

            # Ensure the upload directory exists
            os.makedirs(upload_folder, exist_ok=True)

            image.save(file_path)
            print(f"File saved to {file_path}")

            pred = prediction(file_path)
            if pred is None:
                return jsonify({'error': 'Prediction failed'}), 500

            title = disease_info['disease_name'][pred]
            description = disease_info['description'][pred]
            prevent = disease_info['Possible Steps'][pred]
            image_url = disease_info['image_url'][pred]
            supplement_name = supplement_info['supplement name'][pred]
            supplement_image_url = supplement_info['supplement image'][pred]
            supplement_buy_link = supplement_info['buy link'][pred]

            result = {
                'title': title,
                'desc': description,
                'prevent': prevent,
                'image_url': image_url,
                'pred': str(pred),
                'sname': supplement_name,
                'simage': supplement_image_url,
                'buy_link': supplement_buy_link
            }

            print(result)
            return jsonify(result)

        except Exception as e:
            print(f"Error in submit route: {e}")
            return jsonify({'error': str(e)}), 500

    return render_template('submit.html')

@app.route('/market', methods=['GET', 'POST'])
def market():
    return render_template('market.html', 
                           supplement_image=list(supplement_info['supplement image']),
                           supplement_name=list(supplement_info['supplement name']), 
                           disease=list(disease_info['disease_name']), 
                           buy=list(supplement_info['buy link']))

if __name__ == '__main__':
    print(f"PyTorch version: {torch.__version__}")
    app.run(debug=False, host='127.0.0.1')