import os
import pickle
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tt
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
from ultralytics import YOLO
import requests
from groq import Groq
from datetime import datetime

app = Flask(__name__)

# --- CONFIG ---
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- GROQ API (Replace with your Key) ---
GROQ_API_KEY = "gsk_YOUR_ACTUAL_API_KEY_HERE"
client = Groq(api_key=GROQ_API_KEY)

# --------------------------------------------------------------------------
# 1. DEFINE RESNET ARCHITECTURE (Required for loading .pth)
# --------------------------------------------------------------------------
def conv_block(in_ch, out_ch, pool=False):
    layers = [nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)), nn.Flatten(), nn.Linear(512, num_classes))

    def forward(self, xb):
        out = self.conv1(xb); out = self.conv2(out); out = self.res1(out) + out
        out = self.conv3(out); out = self.conv4(out); out = self.res2(out) + out
        return self.classifier(out)

# --------------------------------------------------------------------------
# 2. LOAD MODELS
# --------------------------------------------------------------------------
models = {}

# A. Crop Recommendation
try:
    with open('models/crop_recommendation.pkl', 'rb') as f:
        models['crop'] = pickle.load(f)
    print("✅ Loaded Crop Model")
except: print("❌ Error loading Crop Model")

# B. Yield Prediction
try:
    models['yield'] = joblib.load('models/yield_prediction.pkl')
    print("✅ Loaded Yield Model")
except: print("❌ Error loading Yield Model")

# C. YOLO Detection
try:
    models['yolo'] = YOLO('models/yolo_model.pt')
    print("✅ Loaded YOLO Model")
except: print("❌ Error loading YOLO Model")

# D. Disease Classification (ResNet)
PLANT_CLASSES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites_Two-spotted_spider_mite', 
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 
    'Tomato___healthy'
]

try:
    resnet = ResNet9(3, len(PLANT_CLASSES))
    # map_location='cpu' is crucial if trained on GPU
    state_dict = torch.load('models/plant_disease_model.pth', map_location=torch.device('cpu'))
    resnet.load_state_dict(state_dict)
    resnet.eval()
    models['disease'] = resnet
    print("✅ Loaded ResNet Disease Model")
except Exception as e: 
    print(f"❌ Error loading ResNet Model: {e}")

# --------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# --------------------------------------------------------------------------
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5,
    'papaya': 6, 'orange': 7, 'apple': 8, 'muskmelon': 9,
    'watermelon': 10, 'grapes': 11, 'mango': 12, 'banana': 13,
    'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19,
    'kidneybeans': 20, 'chickpea': 21, 'coffee': 22
}
reverse_crop_dict = {v: k for k, v in crop_dict.items()}

def create_field(name, label, type='number', placeholder=''):
    return {'name': name, 'label': label, 'type': type, 'placeholder': placeholder}

def get_current_season():
    month = datetime.now().month
    if 6 <= month <= 10: return 'Kharif'
    elif 11 <= month <= 3: return 'Rabi'
    else: return 'Whole Year'

def predict_resnet_image(img_path):
    # Transforms must match training (256x256 resizing)
    stats = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = tt.Compose([tt.Resize((256, 256)), tt.ToTensor(), tt.Normalize(*stats)])
    
    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    
    with torch.no_grad():
        output = models['disease'](img_tensor)
        probs = F.softmax(output, dim=1)
        conf, preds = torch.max(probs, dim=1)
    
    return PLANT_CLASSES[preds.item()], conf.item()

def get_auto_data(lat, lon):
    data = {}
    error = None
    try:
        # A. Location (Reverse Geo)
        geo_url = f"https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}"
        headers = {'User-Agent': 'AgriAI_Platform/1.0'}
        geo_res = requests.get(geo_url, headers=headers).json()
        address = geo_res.get('address', {})
        
        data['State_Name'] = address.get('state', '')
        raw_dist = address.get('county', '') or address.get('state_district', '') or address.get('city', '')
        data['District_Name'] = raw_dist.replace(' District', '').strip()
        data['Crop_Year'] = datetime.now().year
        data['Season'] = get_current_season()

        # B. Weather
        climate_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {"latitude": lat, "longitude": lon, "start_date": "2023-01-01", "end_date": "2023-12-31", "daily": ["temperature_2m_mean", "rain_sum"], "timezone": "auto"}
        w_res = requests.get(climate_url, params=params).json()
        temps = [x for x in w_res['daily']['temperature_2m_mean'] if x]
        rains = [x for x in w_res['daily']['rain_sum'] if x]
        data['temperature'] = round(sum(temps) / len(temps), 2)
        data['rainfall'] = round(sum(rains), 2)
        
        h_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=relative_humidity_2m"
        data['humidity'] = requests.get(h_url).json()['current']['relative_humidity_2m']

        # C. Soil
        soil_url = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=phh2o&property=nitrogen&depth=0-5cm&value=mean"
        s_res = requests.get(soil_url).json()
        data['ph'] = s_res['properties']['layers'][0]['depths'][0]['values']['mean'] / 10.0
        n_val = s_res['properties']['layers'][1]['depths'][0]['values']['mean'] 
        n_scaled = min(max(n_val / 5, 10), 140)
        data['N'] = round(n_scaled, 1)
        data['P'] = round(n_scaled * 0.55, 1)
        data['K'] = round(n_scaled * 0.45, 1)

    except Exception as e:
        error = str(e)
        print(f"API Error: {e}")

    return data, error

# --------------------------------------------------------------------------
# 4. ROUTES (ALL ROUTES MUST BE HERE)
# --------------------------------------------------------------------------

@app.route('/')
def home():
    return render_template('index.html')

# --- 1. CHATBOT ---
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message')
    language = data.get('language', 'English')
    system_prompt = f"You are AgriBot. Answer farming questions in {language}."
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}],
            model="llama-3.3-70b-versatile", temperature=0.7, max_tokens=300
        )
        return {"reply": chat_completion.choices[0].message.content}
    except Exception as e: return {"reply": "Error connecting to AI."}, 500

# --- 2. API ---
@app.route('/api/get_data_by_location', methods=['POST'])
def api_get_data_by_location():
    req = request.get_json()
    data, error = get_auto_data(req.get('lat'), req.get('lon'))
    if error: return {'error': error}, 400
    return data

# --- 3. CROP RECOMMENDATION ---
@app.route('/recommendation', methods=['GET', 'POST'])
def recommendation():
    form_fields = [create_field('N', 'Nitrogen'), create_field('P', 'Phosphorus'), create_field('K', 'Potassium'), create_field('temperature', 'Temperature'), create_field('humidity', 'Humidity'), create_field('ph', 'pH'), create_field('rainfall', 'Rainfall')]
    prediction = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form[f['name']]) for f in form_fields]
            bundle = models['crop']
            sc, mx, model = bundle['sc'], bundle['mx'], bundle['model']
            scaled = sc.transform(mx.transform([inputs]))
            pred_idx = model.predict(scaled)[0]
            prediction = reverse_crop_dict.get(pred_idx, "Unknown").capitalize()
        except Exception as e: prediction = f"Error: {e}"
    return render_template('form.html', title='Crop Recommendation', form_fields=form_fields, prediction=prediction)

# --- 4. YIELD PREDICTION (Using Dropdowns) ---
@app.route('/yield', methods=['GET', 'POST'])
def yield_prediction():
    prediction = None
    bundle = models.get('yield')
    if not bundle: return "Error: Yield Model not loaded."
    
    mappings = bundle['mappings']
    # Lists for Dropdowns
    lists = {
        'states': sorted(list(mappings['state'].keys())),
        'districts': sorted(list(mappings['district'].keys())),
        'seasons': sorted(list(mappings['season'].keys())),
        'crops': sorted(list(mappings['crop'].keys()))
    }

    if request.method == 'POST':
        try:
            # Direct dictionary lookup since we use Dropdowns now
            data = {
                'State_Name': mappings['state'][request.form['State_Name']],
                'District_Name': mappings['district'][request.form['District_Name']],
                'Crop_Year': int(request.form['Crop_Year']),
                'Season': mappings['season'][request.form['Season']],
                'Crop': mappings['crop'][request.form['Crop']],
                'Area': float(request.form['Area'])
            }
            pred = bundle['model'].predict(pd.DataFrame([data]))[0]
            prediction = f"{pred:.2f} Yield/Unit"
        except Exception as e: prediction = f"Error: {e}"

    return render_template('yield_form.html', title='Yield Prediction', prediction=prediction, **lists)

# --- 5. DISEASE CLASSIFIER (RESNET) ---
@app.route('/disease', methods=['GET', 'POST'])
def disease():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            if models.get('disease'):
                try:
                    label, conf = predict_resnet_image(path)
                    res_text = f"{label.replace('___', ' - ').replace('_', ' ')} ({conf*100:.1f}%)"
                    return render_template('yolo.html', title='Disease Classification', original=f'uploads/{filename}', result=res_text, type='text')
                except Exception as e: return f"Error analyzing image: {e}"
            else:
                return "ResNet Model not loaded."
                
    return render_template('yolo.html', title='Disease Classification')

# --- 6. YOLO DETECTION ---
@app.route('/yolo', methods=['GET', 'POST'])
def yolo():
    if request.method == 'POST':
        file = request.files.get('image')
        if file:
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            
            if models.get('yolo'):
                try:
                    results = models['yolo'](path)
                    res_name = 'pred_' + filename
                    Image.fromarray(results[0].plot()[..., ::-1]).save(os.path.join(app.config['UPLOAD_FOLDER'], res_name))
                    return render_template('yolo.html', title='YOLO Detection', original=f'uploads/{filename}', result=f'uploads/{res_name}', type='image')
                except Exception as e: return f"Error analyzing image: {e}"
            else:
                return "YOLO Model not loaded."

    return render_template('yolo.html', title='YOLO Detection')

if __name__ == '__main__':
    app.run(debug=True)