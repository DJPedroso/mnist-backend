"""
MNIST Digit Classifier — Flask API Backend
ICT 120 · BSCS 3A

Endpoints:
  POST /predict   — accepts base64 image, returns predicted digit + probabilities
  GET  /validate  — runs 10-sample validation suite, returns metrics + confusion matrix
  GET  /health    — health check
"""

import os
import io
import base64
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps, ImageFilter

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ── App setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)  # allow requests from Vercel frontend

MODEL_PATH = 'mnist_baseline_model.keras'
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ── Load / train model at startup ────────────────────────────────────────────
print("Loading MNIST data and model...")

(X_train_raw, y_train_raw), (X_test_raw, y_test_raw) = mnist.load_data()
X_train = X_train_raw.reshape(-1, 784) / 255.0
X_test  = X_test_raw.reshape(-1, 784)  / 255.0
y_train_enc = to_categorical(y_train_raw, 10)

if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
    print("Model loaded from disk.")
else:
    print("Training model from scratch (~30-60s)...")
    model = Sequential([
        Dense(128, activation='relu', input_shape=(784,), name='hidden_1'),
        Dense(64,  activation='relu',                     name='hidden_2'),
        Dense(10,  activation='softmax',                  name='output'),
    ], name='MNIST_MLP_Baseline')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train_enc, epochs=10, batch_size=128,
              validation_split=0.1, verbose=1)
    model.save(MODEL_PATH)
    print("Model trained and saved.")

# Full test accuracy
_, test_acc = model.evaluate(
    X_test, to_categorical(y_test_raw, 10), verbose=0
)
print(f"Test accuracy: {test_acc*100:.2f}%")


# ── Image preprocessing ───────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert('L')
    arr = np.array(img)
    if arr.mean() > 127:
        img = ImageOps.invert(img)
    img = img.resize((28, 28), Image.LANCZOS)
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr.reshape(1, 784)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'test_accuracy': round(float(test_acc) * 100, 2)})


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'Missing image field'}), 400

    try:
        # Strip data URI prefix if present
        img_data = data['image']
        if ',' in img_data:
            img_data = img_data.split(',')[1]

        img_bytes = base64.b64decode(img_data)
        img = Image.open(io.BytesIO(img_bytes))
        x = preprocess_image(img)

        probs = model.predict(x, verbose=0)[0]
        predicted = int(np.argmax(probs))
        confidence = float(probs[predicted])

        # Also return 28x28 preview as base64
        preview_arr = (x.reshape(28, 28) * 255).astype(np.uint8)
        preview_img = Image.fromarray(preview_arr).resize((112, 112), Image.NEAREST)
        preview_buf = io.BytesIO()
        preview_img.save(preview_buf, format='PNG')
        preview_b64 = base64.b64encode(preview_buf.getvalue()).decode()

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
        selected_indices, selected_labels, images_b64 = [], [], []

        for digit in range(10):
            candidates = np.where(y_test_raw == digit)[0]
            idx = candidates[digit * 37 % len(candidates)]
            selected_indices.append(idx)
            selected_labels.append(digit)

            # Encode thumbnail as base64
            thumb = (X_test[idx].reshape(28, 28) * 255).astype(np.uint8)
            thumb_img = Image.fromarray(thumb).resize((56, 56), Image.NEAREST)
            buf = io.BytesIO()
            thumb_img.save(buf, format='PNG')
            images_b64.append('data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode())

        X_val = X_test[selected_indices]
        probs  = model.predict(X_val, verbose=0)
        y_pred = np.argmax(probs, axis=1).tolist()
        y_true = selected_labels

        cm = confusion_matrix(y_true, y_pred, labels=list(range(10))).tolist()
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
                'thumbnail':  images_b64[i],
            })

        return jsonify({
            'run_details':     run_details,
            'confusion_matrix': cm,
            'overall_acc':     round(overall * 100, 2),
            'macro_precision': round(macro_p  * 100, 2),
            'macro_recall':    round(macro_r  * 100, 2),
            'macro_f1':        round(macro_f1 * 100, 2),
            'test_accuracy':   round(float(test_acc) * 100, 2),
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
