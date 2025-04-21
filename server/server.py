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
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight


app = FastAPI()


# CORS middleware
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


# Weighted loss function
def weighted_sparse_categorical_crossentropy(weights):
    weights = tf.cast(weights, tf.float32)
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        weights_tensor = tf.gather(weights, y_true)
        unweighted_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
        return unweighted_loss * weights_tensor
    return loss


# Load model and encoders
MODEL_PATH = "models/cnn_lstm_best_daio.h5"
DIRECTION_ENCODER_PATH = "models/direction_encoder.pkl"
BEHAVIOUR_ENCODER_PATH = "models/behaviour_encoder.pkl"


try:
    with open(DIRECTION_ENCODER_PATH, "rb") as f:
        direction_encoder = pickle.load(f)
    with open(BEHAVIOUR_ENCODER_PATH, "rb") as f:
        behaviour_encoder = pickle.load(f)
    print("Encoders loaded successfully.")
except Exception as e:
    print("Error loading encoders:", str(e))
    direction_encoder = None
    behaviour_encoder = None


try:
    model = load_model(MODEL_PATH,custom_objects={
        'loss':weighted_sparse_categorical_crossentropy,
        'weighted_sparse_categorical_crossentropy': weighted_sparse_categorical_crossentropy,
    })
    print("Model loaded successfully.")
    tf.debugging.set_log_device_placement(True)
except Exception as e:
    print("Error loading model:", str(e))
    model = None

def clear_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

def preprocess_csv(csv_path):
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")


    df = df.iloc[:, 2:]
    cols_to_convert = ['Left Pupil X', 'Left Pupil Y', 'Right Pupil X', 'Right Pupil Y',
                       'K Value', 'Fixation Level', 'Saccade Level']
    existing_cols = [col for col in cols_to_convert if col in df.columns]


    if not existing_cols:
        raise HTTPException(status_code=400, detail="No valid numeric columns found in CSV.")


    # Replace empty strings and invalid entries with NaN
    df = df.replace('', np.nan).replace([np.inf, -np.inf], np.nan)


    # Check for columns with all NaN values
    for col in existing_cols:
        if df[col].isna().all():
            raise HTTPException(status_code=400, detail=f"Column {col} contains only NaN values.")


    # Convert to numeric and handle invalid values
    for col in existing_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        # Fill NaN with median, or 0 if median is NaN
        median_val = df[col].median()
        if np.isnan(median_val):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna(median_val)
        # Interpolate remaining NaNs
        df[col] = df[col].interpolate(method='linear', limit_direction='both', limit=10)
        # If any NaNs remain after interpolation, fill with 0
        df[col] = df[col].fillna(0)


    # Scale features
    if existing_cols:
        scaler = StandardScaler()
        # Ensure no constant columns (std = 0)
        for col in existing_cols:
            if df[col].std() == 0:
                df[col] = 0  # Replace constant column with zeros to avoid scaling issues
        df[existing_cols] = scaler.fit_transform(df[existing_cols])


    # Encode categorical variables
    if 'Direction' in df.columns and direction_encoder:
        df['Direction'] = df['Direction'].astype(str)
        valid_directions = df['Direction'].isin(direction_encoder.classes_)
        df.loc[~valid_directions, 'Direction'] = direction_encoder.classes_[0]
        df['Direction'] = direction_encoder.transform(df['Direction'])
    else:
        df['Direction'] = 0


    if 'Behaviour' in df.columns and behaviour_encoder:
        df['Behaviour'] = df['Behaviour'].astype(str)
        valid_behaviours = df['Behaviour'].isin(behaviour_encoder.classes_)
        df.loc[~valid_behaviours, 'Behaviour'] = behaviour_encoder.classes_[0]
        df['Behaviour'] = behaviour_encoder.transform(df['Behaviour'])
    else:
        df['Behaviour'] = 0


    features = []
    for _, row in df.iterrows():
        feature_row = [float(row.get(col, 0.0)) for col in cols_to_convert]
        feature_row.append(float(row.get('Direction', 0)))
        feature_row.append(float(row.get('Behaviour', 0)))
        features.append(feature_row)


    # Pad or truncate to SEQUENCE_LENGTH
    if len(features) > SEQUENCE_LENGTH:
        features = features[:SEQUENCE_LENGTH]
    elif len(features) < SEQUENCE_LENGTH:
        padding = [[0.0] * (len(cols_to_convert) + 2) for _ in range(SEQUENCE_LENGTH - len(features))]
        features.extend(padding)


    input_data = np.array(features, dtype=np.float32)
    # Final check for invalid values
    if np.any(np.isnan(input_data)) or np.any(np.isinf(input_data)):
        raise HTTPException(status_code=400, detail="Input data contains NaN or inf values after preprocessing.")


    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def sanitize_predictions(predictions):
    sanitized = []
    for pred in predictions:
        pred_array = np.array(pred)
        pred_array = np.nan_to_num(pred_array, nan=0.0, posinf=1.0, neginf=-1.0)
        pred_array = pred_array / np.sum(pred_array, axis=-1, keepdims=True)
        sanitized.append(pred_array.tolist())
    return sanitized


@app.post("/predict/")
async def predict_csv(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")


    try:
        temp_file_path = os.path.join(OUTPUT_DIR, f"temp_{file.filename}")
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(temp_file_path)
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


    csv_filename = file.filename.rsplit(".", 1)[0] + "_data.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)


    try:
        subprocess.run(["python", "data.py", video_path, csv_path], check=True)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Error generating CSV from video: {str(e)}")


    try:
        print(csv_path)
        input_data = preprocess_csv(csv_path)
    
        predictions = model.predict(input_data)
        sanitized_predictions = sanitize_predictions(predictions)
        task_pred = sanitized_predictions[0][0]
        attention_pred = sanitized_predictions[1][0]
       
        task_class = TASKS[np.argmax(task_pred)]
        attention_class = CATEGORIES[np.argmax(attention_pred)]

        print("---- DEBUG INFO ----")
        print("CSV Path:", csv_path)
        print("Input shape:", input_data.shape)
        print("Raw predictions:", predictions)
        print("Sanitized predictions:", sanitized_predictions)
        print("Task:", task_class, task_pred)
        print("Attention:", attention_class, attention_pred)

        clear_folder(UPLOAD_DIR)
        clear_folder(OUTPUT_DIR)

        return {
            "predictions": {
                "task": {
                    "class": task_class,
                    "probabilities": task_pred
                },
                "attention": {
                    "class": attention_class,
                    "probabilities": attention_pred
                }
            }
        }


    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating predictions: {str(e)}")