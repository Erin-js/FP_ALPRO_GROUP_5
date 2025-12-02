import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image, ImageOps
import cv2
import easyocr
import difflib

app = Flask(__name__) 
app.secret_key = 'super_secret_key_for_logolens_session'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'static/uploads' 
UPLOAD_PATH = os.path.join(BASE_DIR, UPLOAD_FOLDER)

MODEL_PATH = os.path.join(BASE_DIR, 'model_logo_csv.h5')
TXT_DB_PATH = os.path.join(BASE_DIR, 'Logos.txt')        

os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL = None
OCR_READER = None
DATABASE_BRANDS = []

def init_system():
    global MODEL, OCR_READER, DATABASE_BRANDS

    print("Memulai Inisialisasi Sistem LogoLens (Mode Cepat)...")

    if os.path.exists(MODEL_PATH):
        try:
            MODEL = tf.keras.models.load_model(MODEL_PATH)
            print("CNN Model (Visual) Loaded.")
        except Exception as e:
            print(f"Error Load CNN: {e}")
    else:
        print("File model_logo_csv.h5 tidak ditemukan!")

    if os.path.exists(TXT_DB_PATH):
        with open(TXT_DB_PATH, 'r') as f:
            DATABASE_BRANDS = [line.strip().lower() for line in f.readlines() if line.strip()]
        print(f"Database Merek Loaded: {len(DATABASE_BRANDS)} brands.")
    else:
        DATABASE_BRANDS = ["adidas", "nike", "gucci", "chanel", "starbucks"]

    print("Menyiapkan OCR Reader...")
    try:
        OCR_READER = easyocr.Reader(['en'], gpu=False)
        print("OCR Engine Siap!")
    except Exception as e:
        print(f"Gagal load OCR: {e}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def enhance_image_fast(cv2_img):
    h, w = cv2_img.shape[:2]
    
    if w < 300 or h < 300:
        scale = 2 
        return cv2.resize(cv2_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    
    return cv2_img

def read_text_from_image(image_path):
    if OCR_READER is None: return ""
    
    img = cv2.imread(image_path)
    if img is None: return ""
    img = enhance_image_fast(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    try:
        results = OCR_READER.readtext(gray, detail=0)
        detected_text = " ".join(results).lower().strip()
        return detected_text
    except:
        return ""

def check_brand_match(detected_text):
    best_match = ""
    highest_score = 0
    
    for brand in DATABASE_BRANDS:
        score = difflib.SequenceMatcher(None, brand, detected_text).ratio()
        if score > highest_score:
            highest_score = score
            best_match = brand
            
    return best_match, highest_score

def get_prediction_smart(image_path):
    img_pil = Image.open(image_path).convert('RGB')
    img_pil = ImageOps.fit(img_pil, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.array(img_pil) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if MODEL:
        pred = MODEL.predict(img_array)
        score_vis = pred[0][0] # 0.0 - 1.0
    else:
        score_vis = 0.5 
    detected_text = read_text_from_image(image_path)
    matched_brand, score_text = check_brand_match(detected_text)
    
    print(f"\n ANALISIS: Visual={score_vis:.2f} | Teks='{detected_text}' | Match='{matched_brand}' ({score_text:.2f})")
    final_label = "FAKE"
    final_score = 0.0

    if score_vis > 0.6:
        if score_text >= 0.8:
            final_label = "REAL / GENUINE"
            final_score = (score_vis * 0.4) + (score_text * 0.6) 

        elif score_text > 0.6:
            final_label = "REAL / GENUINE"
            final_score = score_vis 

        else:
            final_label = "FAKE / PALSU"
            final_score = 1.0 - (score_vis * 0.5)

    else:
        if score_text > 0.8:
            final_label = "REAL / GENUINE"
            final_score = score_text * 0.7 
        
        else:
            final_label = "FAKE / PALSU"
            final_score = 1.0 - (score_vis * score_text)

    confidence_percent = int(min(max(final_score * 100, 0), 100))
    
    return final_label, confidence_percent


@app.route('/', methods=['GET'])
def index():
    session.pop('prediction_result', None)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error="Tidak ada file terpilih.")

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        import random
        unique_filename = f"{random.randint(10000, 99999)}_{filename}"
        
        absolute_save_path = os.path.join(UPLOAD_PATH, unique_filename)
        file.save(absolute_save_path)

        label, score = get_prediction_smart(absolute_save_path)

        session['prediction_result'] = {
            'label': label,
            'score': score,
            'image_url': url_for('static', filename=f'uploads/{unique_filename}'),
            'model': 'Hybrid CNN + OCR' 
        }
        
        return redirect(url_for('result'))
    else:
        return render_template('index.html', error="Format file salah.")

@app.route('/result', methods=['GET'])
def result():
    if 'prediction_result' not in session: return redirect(url_for('index'))
    return render_template('result.html', result=session['prediction_result'])

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

if __name__ == '__main__': 
    init_system()
    
    print("\n" + "="*50)
    print("MODE JARINGAN LOKAL (LAN/WIFI)")
    print("="*50)

    try:
        port_input = input("Masukkan Port yang diinginkan (contoh: 5000, 8080, 8000): ")
        port_number = int(port_input)
    except ValueError:
        print("Input salah! Menggunakan port default 5000.")
        port_number = 5000

    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)

    print(f"\n Server Berjalan!")
    print(f"Akses di Laptop ini : http://127.0.0.1:{port_number}")
    print(f"Akses di HP/Teman   : http://{local_ip}:{port_number}")
    print("="*50 + "\n")

    app.run(host='0.0.0.0', port=port_number, debug=True)
