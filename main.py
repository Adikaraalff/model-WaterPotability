import pandas as pd
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import Flask-CORS

# Membuat aplikasi Flask
app = Flask(__name__)

# Aktifkan CORS
CORS(app)  # Mengizinkan semua domain

# Muat model yang sudah disimpan
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Endpoint untuk halaman home /
@app.route('/')
def welcome():
    return "<h1>Selamat Datang di API Water Potability</h1>"

@app.route('/predict', methods=['POST'])
def predict_water_potability():
    try:
        # Ambil data dari request
        data = request.get_json()

        # Buat DataFrame dengan nama kolom yang sesuai dengan fitur dataset water potability
        input_data = pd.DataFrame([{
            "ph": data['ph'],
            "Hardness": data['Hardness'],
            "Solids": data['Solids'],
            "Chloramines": data['Chloramines'],
            "Sulfate": data['Sulfate'],
            "Conductivity": data['Conductivity'],
            "Organic_carbon": data['Organic_carbon'],
            "Trihalomethanes": data['Trihalomethanes'],
            "Turbidity": data['Turbidity']
        }])

        # Melakukan prediksi dengan model yang sudah dimuat
        prediction = model.predict(input_data)

        # Mendapatkan probabilitas prediksi
        probabilities = model.predict_proba(input_data)

        # Probabilitas untuk kelas positif dan negatif
        probability_negative = probabilities[0][0] * 100  # Probabilitas untuk kelas 0 (non-potable)
        probability_positive = probabilities[0][1] * 100  # Probabilitas untuk kelas 1 (potable)

        # Prediksi output (0 atau 1, di mana 1 berarti air bisa diminum)
        if prediction[0] == 1:
            result = f"Air ini dapat diminum. Kemungkinan air dapat diminum adalah {probability_positive:.2f}%."
        else:
            result = f"Air ini tidak dapat diminum. Kemungkinan air tidak dapat diminum adalah {probability_negative:.2f}%. "

        # Kembalikan hasil prediksi dan probabilitas dalam bentuk JSON
        return jsonify({
            'prediction': result,
            'probabilities': {
                'negative': f"{probability_negative:.2f}%",  # Format 2 desimal
                'positive': f"{probability_positive:.2f}%"   # Format 2 desimal
            }
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

# Jalankan aplikasi Flask
if __name__ == '__main__':
    app.run(debug=True)
