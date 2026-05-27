"""
MNIST Digit Classifier — Flask API Backend
ICT 120 · BSCS 3A
 
Loads a pre-trained Keras model (mnist_baseline_model.keras) directly.
No training needed — fast startup, low memory.
"""
 
import os, io, base64
import numpy as np
import gc
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
 
app = Flask(__name__)
CORS(app)
 
MODEL_PATH = 'mnist_baseline_model.keras'
 
# Globals — populated after app starts (so gunicorn binds the port first)
model      = None
X_test     = None
y_test_raw = None
test_acc   = None
 
 
def load_resources():
    global model, X_test, y_test_raw, test_acc

    print("Loading model...")
    model = load_model(MODEL_PATH)
    print("Model loaded!")

    print("Loading MNIST test data...")
    (_, _), (X_test_raw_local, y_test_raw_local) = mnist.load_data()
    X_test     = X_test_raw_local.reshape(-1, 784) / 255.0
    y_test_raw = y_test_raw_local
    print("Ready.")

    # Reduce sample size from 1000 to 100
    y_pred_all = np.argmax(model.predict(X_test[:100], verbose=0), axis=1)
    test_acc   = float(np.mean(y_pred_all == y_test_raw[:100]))
    print(f"Accuracy (100 sample): {test_acc*100:.2f}%")
    gc.collect()

# Load inside app context so gunicorn can bind the port before heavy work starts
with app.app_context():
    load_resources()
 
 
def preprocess_image(img: Image.Image) -> np.ndarray:
    if img.mode == 'RGBA':
        bg = Image.new('RGBA', img.size, (0, 0, 0, 255))
        img = Image.alpha_composite(bg, img).convert('L')
    else:
        img = img.convert('L')
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    # Resize FIRST, then enhance strokes on the small image
    img = img.resize((28, 28), Image.LANCZOS)
    img = img.filter(ImageFilter.MaxFilter(3))       # ← moved after resize
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 784)
 
 
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'test_accuracy': round(test_acc * 100, 2)})
 
 
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400
    try:
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]
        img = Image.open(io.BytesIO(base64.b64decode(img_data)))
        x = preprocess_image(img)
 
        probs = model.predict(x, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])
 
        preview_arr = (x.reshape(28, 28) * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview_arr).resize((112, 112), Image.NEAREST)
        buf = io.BytesIO()
        preview_img.save(buf, format='PNG')
        preview_b64 = base64.b64encode(buf.getvalue()).decode()
 
        return jsonify({
            'predicted': predicted,
            'confidence': round(confidence * 100, 2),
            'probabilities': [round(float(p) * 100, 2) for p in probs],
            'preview': f'data:image/png;base64,{preview_b64}',
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
@app.route('/validate', methods=['GET'])
def validate():
    try:
        indices, labels, thumbs = [], [], []
        for digit in range(10):
            candidates = np.where(y_test_raw == digit)[0]
            idx = candidates[digit * 37 % len(candidates)]
            indices.append(idx)
            labels.append(digit)
            thumb = (X_test[idx].reshape(28, 28) * 255).astype(np.uint8)
            thumb_img = Image.fromarray(thumb).resize((56, 56), Image.NEAREST)
            buf = io.BytesIO()
            thumb_img.save(buf, format='PNG')
            thumbs.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())
 
        X_val = X_test[indices]
        probs = model.predict(X_val, verbose=0)
        y_pred = np.argmax(probs, axis=1).tolist()
        y_true = labels
 
        cm       = confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist()
        overall  = float(accuracy_score(y_true, y_pred))
        macro_p  = float(precision_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(10))))
        macro_r  = float(recall_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(10))))
        macro_f1 = float(f1_score(y_true, y_pred, average='macro', zero_division=0, labels=list(range(10))))
 
        run_details = []
        for i in range(10):
            run_details.append({
                'run': i + 1,
                'true_label': int(y_true[i]),
                'predicted':  int(y_pred[i]),
                'confidence': round(float(probs[i, y_pred[i]]) * 100, 2),
                'correct':    bool(y_pred[i] == y_true[i]),
                'thumbnail':  thumbs[i],
            })
 
        return jsonify({
            'run_details':      run_details,
            'confusion_matrix': cm,
            'overall_acc':      round(overall  * 100, 2),
            'macro_precision':  round(macro_p  * 100, 2),
            'macro_recall':     round(macro_r  * 100, 2),
            'macro_f1':         round(macro_f1 * 100, 2),
            'test_accuracy':    round(test_acc * 100, 2),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500
 
 
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
 
