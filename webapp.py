import os
import random
import numpy as np
# Import Flask dan modul-modul lain yang diperlukan
from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
from PIL import Image

app = Flask(__name__) 
app.secret_key = 'super_secret_key_for_logolens_session'

UPLOAD_FOLDER = 'static/uploads' 
UPLOAD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), UPLOAD_FOLDER)

os.makedirs(UPLOAD_PATH, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

MODEL = None

def allowed_file(filename):
    """Mengecek apakah ekstensi file diizinkan."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_cnn_model():
    """
    Fungsi MOCK untuk memuat model CNN (simulasi).
    """
    global MODEL
    class MockModel:
        def predict(self, processed_image):
            confidence_real = random.uniform(0.80, 0.99)
            confidence_fake = 1.0 - confidence_real
            return np.array([[confidence_fake, confidence_real]])

    MODEL = MockModel()
    print("Menggunakan Mock Model untuk simulasi prediksi.")

def preprocess_image(image_path):
    """
    Pra-pemrosesan gambar: resize, konversi ke NumPy array, dan normalisasi.
    """
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((150, 150))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0) 
        return img_array
    except Exception as e:
        print(f"Error saat memproses gambar: {e}")
        return None

def get_prediction(image_path):
    """Fungsi utama untuk mendapatkan prediksi dari gambar."""
    processed_image = preprocess_image(image_path)
    
    if processed_image is None or MODEL is None:
        return "ERROR", 0, image_path

    # Simulasi prediksi
    predictions = MODEL.predict(processed_image)
    
    # Kelas (FAKE = 0, REAL = 1)
    predicted_class_index = np.argmax(predictions[0])
    
    # Confidence Score
    confidence_score = predictions[0][predicted_class_index] * 100
    confidence_score = int(round(confidence_score))
    
    labels = ["FAKE", "REAL"]
    prediction_label = labels[predicted_class_index]
    
    # Menyesuaikan skor simulasi
    if confidence_score < 80:
        confidence_score = random.randint(80, 95) 

    return prediction_label, confidence_score, image_path


@app.route('/', methods=['GET'])
def index():
    """Halaman utama untuk unggah logo."""
    session.pop('prediction_result', None) 
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Menangani unggahan file dan menjalankan prediksi."""
    if 'file' not in request.files or request.files['file'].filename == '':
        return render_template('index.html', error="Tidak ada file terpilih.")

    file = request.files['file']

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        unique_filename = f"{random.getrandbits(64)}_{filename}"
        
        absolute_save_path = os.path.join(UPLOAD_PATH, unique_filename)
        file.save(absolute_save_path)
        
        prediction_label, confidence_score, _ = get_prediction(absolute_save_path)

        session['prediction_result'] = {
            'label': prediction_label,
            'score': confidence_score,

            'image_url': url_for('static', filename=f'uploads/{unique_filename}'),
            'model': 'CNN'
        }
        
        return redirect(url_for('result'))
    else:
        return render_template('index.html', error="Format file tidak diizinkan. Gunakan PNG, JPG, atau JPEG.")


@app.route('/result', methods=['GET'])
def result():
    """Menampilkan hasil prediksi yang disimpan di session."""
    if 'prediction_result' not in session:
        return redirect(url_for('index'))
    
    result_data = session['prediction_result']
    return render_template('result.html', result=result_data)


@app.route('/about', methods=['GET'])
def about():
    """Halaman informasi aplikasi."""
    return render_template('about.html')


if __name__ == '__main__': 
    load_cnn_model()
    print(">>> Flask server dijalankan...")
    app.run(debug=True, port=8080)