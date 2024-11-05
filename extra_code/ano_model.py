import os
import gc
import sys
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import r2_score
import tensorflow as tf
import logging
import psutil
import subprocess
import matplotlib.pyplot as plt
from tensorflow.keras.mixed_precision import set_global_policy

set_global_policy('mixed_float16')

BATCHSIZE = int(sys.argv[1])
EPOCHS = int(sys.argv[2])
lr = float(sys.argv[3])
fps_file = sys.argv[4]
y_true_file = sys.argv[5]
##################################################################
model_name = sys.argv[6] if len(sys.argv) > 6 else None
target_path = sys.argv[7] if len(sys.argv) > 7 else None
cv = int(sys.argv[8]) if len(sys.argv) > 8 and sys.argv[8] != 'None' else None
test_size = float(sys.argv[9]) if len(sys.argv) > 9 else 0.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def print_cpu_memory():
    memory_info = psutil.virtual_memory()
    logging.info(f"Total Memory: {memory_info.total / (1024 ** 3):.2f} GB")
    logging.info(f"Available Memory: {memory_info.available / (1024 ** 3):.2f} GB")
    logging.info(f"Used Memory: {memory_info.used / (1024 ** 3):.2f} GB")
    logging.info(f"Memory Usage: {memory_info.percent}%")
def print_gpu_memory(status=""):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for idx, line in enumerate(lines):
                used, total = line.split(', ')
                logging.info(f"[{status}] GPU {idx}: Memory Usage: {used} MB / {total} MB")
    except Exception as e:
        logging.error(f"Error executing nvidia-smi: {e}")
def save_history_plot(history, target_path, model_name, test_size, fold=None):
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'Model Loss (test_size={test_size})')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(2, 1, 2)
    for metric in history.history:
        if metric.startswith('val_'):
            continue
        plt.plot(history.history[metric], label=f'Training {metric}')
        val_metric = f'val_{metric}'
        if val_metric in history.history:
            plt.plot(history.history[val_metric], label=f'Validation {metric}')
    
    plt.title(f'Model Metrics (test_size={test_size})')
    plt.ylabel('Value')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.tight_layout()
    
    file_name = f"{model_name}_history{'_fold'+str(fold) if fold else ''}_test_size[{test_size}].png"
    plt.savefig(os.path.join(target_path, model_name, file_name), dpi=300)
    plt.close()
def load_model(target_path, model_name, test_size, cv=None):
    model_path = f"{target_path}/{model_name}/{model_name}_full_model{'_cv'+str(cv) if cv else ''}_test_size[{test_size}].keras"
    try:
        if os.path.exists(model_path):
            model = tf.keras.models.load_model(model_path, compile=False)
            logging.info(f"Model successfully loaded from {model_path}")
            return model
        else:
            logging.error(f"Model path does not exist: {model_path}")
            return None
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        return None
def preprocess_data(xtr, ytr, use_parallel=False):
    dataset = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    if use_parallel:
        dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=len(xtr)).batch(BATCHSIZE).cache().prefetch(tf.data.AUTOTUNE)
    return dataset

# def train_model(model, train_dataset, target_path, model_name, fold=None):
def train_model(model, train_dataset, valid_dataset, target_path, model_name, fold=None):
    checkpoint_dir = f"{target_path}/checkpoints/{model_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model{'_fold'+str(fold) if fold else ''}.keras")
    cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        save_best_only=True,
        monitor='val_loss',
        mode='min',
        verbose=1,
    )    
    # es = tf.keras.callbacks.EarlyStopping(
    #     monitor='val_loss',  
    #     patience=EPOCHS,
    #     restore_best_weights=True,
    #     mode='min',
    #     verbose=0,
    # )
    
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=valid_dataset,
        # callbacks=[cp, es],
        callbacks=[cp], #, es],
        verbose=0,
    )
    save_history_plot(history, target_path, model_name, fold)
    del train_dataset
    gc.collect()
def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()
    logging.info("GPU memory cleared.")
def main():
    try:
        os.makedirs(f"{target_path}/{model_name}", exist_ok=True)
        model = load_model(target_path, model_name, test_size, cv)
        if model is None:
            raise ValueError("Failed to load model")

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.RootMeanSquaredError()
            ]
        )
        
        fps = np.load(fps_file)
        y_true = np.load(y_true_file)

        model_input_shape = model.input_shape
        if model_input_shape[1] != fps.shape[1]:
            raise ValueError(f"Model input dimension ({model_input_shape[1]}) does not match data dimension ({fps.shape[1]})")

        if cv is not None and cv > 1:
            xtr, xte, ytr, yte = train_test_split(fps, y_true, test_size=test_size, random_state=42)
            kf = KFold(n_splits=cv, shuffle=True, random_state=42)
            avg_r2_score = []

            for fold, (train_index, test_index) in enumerate(kf.split(xtr), 1):
                xtr_cv, xte_cv = xtr[train_index], xtr[test_index]
                ytr_cv, yte_cv = ytr[train_index], ytr[test_index]

                train_dataset = preprocess_data(xtr_cv, ytr_cv, use_parallel=True)
                train_model(model, train_dataset, target_path, model_name, fold)

                ypred = model.predict(xte_cv, verbose=0)
                r2_scores = r2_score(yte_cv, ypred)
                
                if np.isnan(r2_scores) or np.isinf(r2_scores) or r2_scores <= 0:
                    logging.warning(f"[cv][{fold}th] : R2 score : 0.000000 (prune)")
                else:
                    logging.info(f"[cv][{fold}th] : R2 score : {r2_scores:.6f}")
                
                avg_r2_score.append(r2_scores)
                clear_gpu_memory()
                print_cpu_memory()
                print_gpu_memory(f"Fold {fold}")
            r2_result_res_avg = np.mean(avg_r2_score)
            logging.info(f"[cv][{fold}th][Avg] : R2 score : {r2_result_res_avg:.6f}")
            ypred = model.predict(xte, verbose=0)            
            r2_result = r2_score(yte, ypred)
            os.makedirs(f"save_model/{model_name}", exist_ok=True)
            model.save(f"save_model/{model_name}/{model_name}_full_model{'_cv'+str(cv) if cv else ''}_test_size[{test_size}]_r2score[{r2_result:<.4f}].keras")
            del model
            logging.info(f"[cv][{fold}th][Result] : R2 score : {r2_result:.6f}")
            print(f"{r2_result:.6f}")
        else:
            xtr, xte, ytr, yte = train_test_split(fps, y_true, test_size=test_size, random_state=42)
            xtr, xtev, ytr, ytev = train_test_split(xtr, ytr, test_size=0.1, random_state=42)
            train_dataset = preprocess_data(xtr, ytr, use_parallel=True)
            valid_dataset = preprocess_data(xtev, ytev, use_parallel=True)
            train_model(model, train_dataset, valid_dataset, target_path, model_name)
            # train_model(model, train_dataset, target_path, model_name)

            ypred = model.predict(xte, verbose=0)
            r2_result = r2_score(yte, ypred)
            
            os.makedirs(f"save_model/{model_name}", exist_ok=True)
            model.save(f"save_model/{model_name}/{model_name}_full_model{'_cv'+str(cv) if cv else ''}_test_size[{test_size}]_r2score[{r2_result:<.4f}].keras")
            del model

            if np.isnan(r2_result) or np.isinf(r2_result) or r2_result <= 0:
                logging.warning("R2: 0.000000 (prune)")
            else:
                logging.info(f"R2: {r2_result:.6f}")
            print(f"{r2_result:.6f}")

    except Exception as e:
        logging.error(f"Error in learning process: {e}")
        print("0.000000")

    finally:
        clear_gpu_memory()
        print_cpu_memory()
        print_gpu_memory("Final")

if __name__ == "__main__":
    main()
