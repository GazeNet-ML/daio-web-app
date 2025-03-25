from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import tensorflow as tf
import numpy as np
import os
import pickle
import shutil
import subprocess
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model

app = FastAPI()

# needed to access from any port
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CATEGORIES = ["BVPS (GTSS)", "BVPS (TSS)", "GVPS (BTSS)", "GVPS (TSS)"]
TASKS = ["Picture", "Reading", "Video"]
SEQUENCE_LENGTH = 30

UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# added loss function from train.py
def weighted_sparse_categorical_crossentropy(weights):
    weights = tf.cast(weights, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights_tensor = tf.gather(weights, y_true)
        unweighted_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return unweighted_loss * weights_tensor
    return loss

# constant weights for now need to change
task_weights = tf.constant([0.5, 2.0, 1.5, 1.0], dtype=tf.float32)
attention_weights = tf.constant([1.0, 1.2, 0.8, 1.5], dtype=tf.float32)

# Load model (final and best bot have kinda same performance)
MODEL_PATH = "models/cnn_lstm_final.h5"
DIRECTION_ENCODER_PATH = "models/direction_encoder.pkl"
BEHAVIOUR_ENCODER_PATH = "models/behaviour_encoder.pkl"

try:
    model = load_model(MODEL_PATH, custom_objects={
        'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy,
        'loss': weighted_sparse_categorical_crossentropy(task_weights)
    })
    
    model.compile(
        optimizer="adam",
        loss={
            'task_output': weighted_sparse_categorical_crossentropy(task_weights),
            'attention_output': weighted_sparse_categorical_crossentropy(attention_weights)
        },
        metrics=['accuracy']
    )
    print("✅ Model loaded successfully.")

except Exception as e:
    print("Error loading model:", str(e))
    model = None

# Load encoders
with open(DIRECTION_ENCODER_PATH, "rb") as f:
    direction_encoder = pickle.load(f)

with open(BEHAVIOUR_ENCODER_PATH, "rb") as f:
    behaviour_encoder = pickle.load(f)


def preprocess_csv(csv_path):
    #pretty much copy of train.py
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    df = df.iloc[:, 2:]

    cols_to_convert = ['Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y',
                       'K Value', 'Fixation Level', 'Saccade Level']
    
    existing_cols = [col for col in cols_to_convert if col in df.columns]

    # Replace missing value indicator and convert to numeric
    df = df.replace('', np.nan)
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Enhanced missing value handling
    for col in existing_cols:
        # Fill NaNs with median
        df[col] = df[col].fillna(df[col].median())
        
        # Interpolate with linear method
        df[col] = df[col].interpolate(method='linear', limit_direction='both')
    
    # Replace infinite values
    df[existing_cols] = df[existing_cols].replace([np.inf, -np.inf], 1e6)

    # Categorical encoding (if encoders are loaded)
    if 'Direction' in df.columns:
        df['Direction'] = direction_encoder.transform(df['Direction'].astype(str))
    if 'Behaviour' in df.columns:
        df['Behaviour'] = behaviour_encoder.transform(df['Behaviour'].astype(str))

    features = []
    for _, row in df.iterrows():
        feature_row = [
            float(row.get(col, 0.0)) for col in cols_to_convert
        ]
        
        feature_row.append(row.get('Direction', 0))
        feature_row.append(row.get('Behaviour', 0))
        
        features.append(feature_row)

    # Ensure sequence length is exactly 30
    if len(features) > SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]
    elif len(features) < SEQUENCE_LENGTH:
        # Pad with zero vectors
        padding = [[0.0] * (len(cols_to_convert) + 2) for _ in range(SEQUENCE_LENGTH - len(features))]
        features.extend(padding)

    # Convert to numpy array and add batch dimension
    input_data = np.array(features, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    return input_data

#need this coz NaN values are annoying 
def sanitize_predictions(predictions):

    sanitized = []
    for pred in predictions:
        # Ensure numpy array and replace NaNs
        pred_array = np.array(pred)
        pred_array = np.nan_to_num(pred_array, 
                                   nan=0.0,   # Replace NaNs with 0
                                   posinf=1.0,  # Replace +inf with 1
                                   neginf=-1.0)  # Replace -inf with -1
        
        # Ensure probabilities sum to 1 for softmax outputs
        pred_array = pred_array / np.sum(pred_array, axis=-1, keepdims=True)
        
        sanitized.append(pred_array.tolist())
    
    return sanitized

@app.post("/predict/")
async def predict_csv(file: UploadFile = File(...)):
    try:
        temp_file_path = os.path.join(OUTPUT_DIR, f"temp_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        input_data = preprocess_csv(temp_file_path)
        predictions = model.predict(input_data)
        sanitized_output = sanitize_predictions(predictions)

        os.remove(temp_file_path)
        return {"predictions": sanitized_output}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")


@app.post("/upload/")
async def upload_video(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    video_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Convert video to CSV using external script
    csv_filename = file.filename.rsplit(".", 1)[0] + "_data.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)

    try:
        subprocess.run(["python", "data.py", video_path, csv_path], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV from video: {str(e)}")

    # Process the generated CSV and make predictions
    try:
        input_data = preprocess_csv(csv_path)
        predictions = model.predict(input_data)
        
        # Decode predictions
        task_pred = predictions[0][0]  
        attention_pred = predictions[1][0]  
        
        # Get performed task
        task_class = TASKS[np.argmax(task_pred)]
        attention_class = CATEGORIES[np.argmax(attention_pred)]
        
        return {
            "predictions": {
                "task": {
                    "class": task_class,
                    "probabilities": task_pred.tolist()
                },
                "attention": {
                    "class": attention_class,
                    "probabilities": attention_pred.tolist()
                }
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")