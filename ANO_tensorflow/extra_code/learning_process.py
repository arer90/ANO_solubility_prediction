import os
import gc
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import logging

from tensorflow.keras.mixed_precision import set_global_policy
set_global_policy('mixed_float16')

BATCHSIZE = int(sys.argv[1])
EPOCHS = int(sys.argv[2])
lr = float(sys.argv[3])
fps_file = sys.argv[4]
y_true_file = sys.argv[5]
trial_number = int(sys.argv[6]) if len(sys.argv) > 6 else None

def save_history_plot(history):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.title(f'Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    for metric in history.history:
        if metric != 'loss':
            plt.plot(history.history[metric], label=metric)
    plt.title(f'Model Metrics')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"save_model/full_model.png", dpi=300)
    plt.close()

def load_model():
    model_path = "save_model/full_model.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        logging.info(f"Model successfully loaded from {model_path}")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None

def preprocess_data(xtr, ytr):
    dataset = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    dataset = dataset.shuffle(buffer_size=len(xtr)).batch(BATCHSIZE).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

def train_model(model, train_dataset, valid_dataset):
    cb = []
    if trial_number is not None:
        class ReportIntermediateCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if logs and 'val_loss' in logs:
                    print(f"intermediate_value:{epoch}:{-logs['val_loss']}")
                    sys.stdout.flush()
        cb.append(ReportIntermediateCallback())
    
    cb.append(
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=50,
            restore_best_weights=True,
            mode='min',
            verbose=1
        )
    )
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        callbacks=cb,
        verbose=0
    )
    save_history_plot(history)
    return history

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()
    logging.info("GPU memory cleared.")

def main():
    try:
        model = load_model()
        if model is None:
            raise ValueError("Failed to load model")

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                      loss=tf.keras.losses.MeanSquaredError(),
                      metrics=[tf.keras.losses.MeanSquaredError(),
                               tf.keras.losses.MeanAbsoluteError(),
                               tf.keras.metrics.RootMeanSquaredError()])

        fps = np.load(fps_file)
        y_true = np.load(y_true_file)

        model_input_shape = model.input_shape
        if model_input_shape[1] != fps.shape[1]:
            raise ValueError(f"Model input dimension ({model_input_shape[1]}) does not match data dimension ({fps.shape[1]})")

        xtr, xte, ytr, yte = train_test_split(fps, y_true, test_size=0.2, random_state=42)
        xtr, xtev, ytr, ytev = train_test_split(xtr, ytr, test_size=0.1, random_state=42)
        train_dataset = preprocess_data(xtr, ytr)
        valid_dataset = preprocess_data(xtev, ytev)
        
        train_model(model, train_dataset, valid_dataset)

        ypred = model.predict(xte, verbose=0)
        
        if np.any(np.isnan(ypred)) or np.any(np.isinf(ypred)):
            raise ValueError("Invalid predictions: NaN or inf values encountered.")

        r2_result = r2_score(yte, ypred)
        
        if np.isnan(r2_result) or np.isinf(r2_result) or r2_result <= 0:
            print("R2: 0.0 (prune)")
        else:
            print(f"R2: {r2_result:.6f}")
        
    except Exception as e:
        logging.error(f"Error in learning process: {e}")
        print("0.000000")

    finally:
        clear_gpu_memory()

if __name__ == "__main__":
    main()
