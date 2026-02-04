# 🌱 AgriAI: Smart Farming Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Framework-Flask-green?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/AI-PyTorch-orange?style=for-the-badge&logo=pytorch&logoColor=white)
![YOLOv8](https://img.shields.io/badge/Vision-YOLOv8-yellow?style=for-the-badge)
![Llama 3](https://img.shields.io/badge/GenAI-Llama%203-purple?style=for-the-badge)

> **Empowering farmers with Artificial Intelligence, Computer Vision, and Generative AI.**

**AgriAI** is a comprehensive full-stack web application designed to assist farmers in making data-driven decisions. It integrates predictive machine learning models, deep learning for disease detection, and Large Language Models (LLMs) to provide a holistic farming assistant.

---

## 🚀 Key Features

### 🧠 **1. AI-Driven Crop Intelligence**
* **🌱 Crop Recommendation:** Analyzes Soil parameters (N-P-K, pH) and Weather conditions to suggest the most suitable crop using a **Random Forest** algorithm.
* **📊 Yield Prediction:** Estimates agricultural production output based on Location (State/District), Season, and historical data.

### 👁️ **2. Computer Vision (Disease Diagnosis)**
* **🦠 Disease Classification:** Identifies plant diseases from leaf images with high accuracy using a **ResNet9** Deep Learning model.
* **🎯 YOLO Object Detection:** Detects and draws bounding boxes around infected areas on leaves using **YOLOv8**.

### 🤖 **3. Generative AI Assistant**
* **💬 AgriBot:** A 24/7 intelligent chatbot powered by **Llama 3 (via Groq API)**.
* **🗣️ Multilingual Support:** Answers queries in **English, Hindi, Telugu, and Tamil** to assist farmers in their local language.

### 📍 **4. Smart Automation**
* **🌍 Location-Aware:** Uses **GPS** to automatically fetch real-time weather (Open-Meteo) and soil data (SoilGrids).
* **⚡ Zero-Typing Forms:** Auto-fills State, District, and Season based on the user's live location.

---

## 🛠️ Tech Stack

| Domain | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Tailwind CSS, JavaScript (Glassmorphism UI) |
| **Backend** | Flask (Python) |
| **Machine Learning** | Scikit-Learn (Random Forest), Pandas, NumPy |
| **Deep Learning** | PyTorch (ResNet9), Ultralytics (YOLOv8) |
| **GenAI** | Groq API (Llama-3-70b-Versatile) |
| **APIs** | Open-Meteo (Weather), SoilGrids, OpenStreetMap (Geocoding) |

---

## 📂 Project Structure

```bash
AgriAI/
├── app.py                   # Main Flask Application
├── requirements.txt         # Project Dependencies
├── models/                  # AI Models (.pkl, .pt, .pth)
├── training_code/           # Original Python Scripts for Model Training
├── static/
│   ├── css/                 # Custom Styles (Glassmorphism)
│   └── uploads/             # User uploaded images
└── templates/               # HTML Pages
    ├── index.html           # Dashboard
    ├── form.html            # Smart Forms (Crop/Yield)
    ├── yolo.html            # Disease Results
    └── base.html            # Layout & Chatbot Widget
```

## 📷 Screenshots
<img width="1470" height="798" alt="Screenshot 2026-02-04 at 7 44 43 PM" src="https://github.com/user-attachments/assets/c6681933-5fb9-4be0-86ba-008d5ca7feb6" />
<img width="2940" height="1596" alt="image" src="https://github.com/user-attachments/assets/203ce003-789d-43bc-a963-33b1ac3cbe3c" />
<img width="1470" height="797" alt="Screenshot 2026-02-04 at 7 47 36 PM" src="https://github.com/user-attachments/assets/9daf7df3-08ad-4b6a-b69b-1aebacde27af" />
<img width="1469" height="336" alt="Screenshot 2026-02-04 at 7 47 50 PM" src="https://github.com/user-attachments/assets/ef98aa56-ca64-4a67-99b8-89b817fb3282" />
<img width="1470" height="798" alt="Screenshot 2026-02-04 at 7 48 24 PM" src="https://github.com/user-attachments/assets/fadc6daa-fd66-4ac3-90f0-ae9b33dc54ab" />
<img width="1468" height="798" alt="Screenshot 2026-02-04 at 7 48 59 PM" src="https://github.com/user-attachments/assets/9bf54055-c125-45ad-bff6-4505aa71f800" />
<img width="1470" height="796" alt="Screenshot 2026-02-04 at 7 50 25 PM" src="https://github.com/user-attachments/assets/e3a9de9b-9457-4bb5-abf2-5d8f9d96ea7b" />
<img width="1469" height="797" alt="Screenshot 2026-02-04 at 7 59 54 PM" src="https://github.com/user-attachments/assets/9462b865-5b04-444f-a2f7-b8f8829f2b26" />

---

## ⚙️ Installation & Setup

Follow these steps to set up the project locally.

### **1. Clone the Repository**
```bash
git clone [https://github.com/Lasyareddy26/AgriAI.git](https://github.com/Lasyareddy26/AgriAI.git)
cd AgriAI
```

### 2. Install Dependencies
```bash
# Create virtual environment (Optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

# Install required packages
pip install -r requirements.txt
```

### 3. Set Up API Keys
```bash
GROQ_API_KEY = "gsk_your_actual_api_key_here"
```

### 4. Run the Application
```bash
python app.py
```

---
## 📜 License
This project is open-source and available under the MIT License.

