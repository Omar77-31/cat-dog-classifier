# 🐱🐶 Cat vs Dog Image Classifier

A Django web application that uses a Convolutional Neural Network (CNN) to classify uploaded images as either a Cat or a Dog.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Django](https://img.shields.io/badge/Django-5.2-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)

## 📌 Features

- **Web Interface:** Upload images directly from the browser.
- **Image Preprocessing:** Resizes, normalizes, and validates uploaded images.
- **Neural Network Inference:** Runs a forward pass through a custom CNN or a pre-trained MobileNetV2 model.
- **Prediction History:** Stores the last 10 predictions in an SQLite database.
- **Admin Panel:** Manage predictions via Django's built-in admin.
- **Security:** Uses environment variables for sensitive settings.

## 🛠️ Tech Stack

- **Backend:** Django 5.2
- **Machine Learning:** TensorFlow / Keras
- **Image Processing:** Pillow, NumPy
- **Database:** SQLite
- **Frontend:** HTML, CSS (Django Templates)

## 📂 Project Structure
nn_project/
├── nn_project/ # Main Django configuration
├── object_detection/ # Main application
│ ├── ml/ # Neural network logic and scripts
│ │ ├── classifier.py # Inference logic (forward pass)
│ │ ├── network.py # CNN architecture definition
│ │ ├── train.py # Model training script
│ │ └── image_fundamentals_demo.py
│ ├── models/ # Database models
│ ├── views/ # Web views (upload, result)
│ └── templates/ # HTML templates
├── manage.py
├── requirements.txt
└── .env.example

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/Omar77-31/cat-dog-classifier.git
cd cat-dog-classifier
###2. Create a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
###3. Install dependencies
pip install -r requirements.txt
###4. Set up environment variables
cp .env.example .env
###5. Apply migrations
python manage.py migrate
###6. Run the development server
python manage.py runserver
Open your browser and go to http://127.0.0.1:8000
