import os
import gc
import sys
import numpy as np
from sklearn.metrics import r2_score, root_mean_squared_error, mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.mixed_precision import set_global_policy

import logging

RANDOM_STATE = 42

def setup_gpu():
    """GPU 설정을 안전하게 수행"""
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            set_global_policy('mixed_float16')
    except Exception as e:
        print(f"GPU setup warning: {e}")
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

setup_gpu()

# --- 파라미터 읽기 ---
BATCHSIZE = int(sys.argv[1])
EPOCHS = int(sys.argv[2])
lr = float(sys.argv[3])
xtr_file = sys.argv[4]
ytr_file = sys.argv[5]
xval_file = sys.argv[6]
yval_file = sys.argv[7]
mode = sys.argv[8] if len(sys.argv) > 8 else "test"  # "cv" or "test"
trial_number = int(sys.argv[9]) if len(sys.argv) > 9 else None

def load_model():
    model_path = "save_model/full_model.keras"
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        return None

def compile_model(model):    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[tf.keras.metrics.MeanAbsoluteError(), 
                 tf.keras.metrics.RootMeanSquaredError()],
    )
    return model

def load_data_optimized():
    try:
        X_tr = np.load(xtr_file).astype(np.float32)
        y_tr = np.load(ytr_file).astype(np.float32)
        X_val = np.load(xval_file).astype(np.float32)
        y_val = np.load(yval_file).astype(np.float32)
        
        # y는 1차원으로 유지
        if len(y_tr.shape) > 1:
            y_tr = y_tr.flatten()
        if len(y_val.shape) > 1:
            y_val = y_val.flatten()
            
    except Exception as e:
        raise e
    
    # X와 y를 따로 검사 (shape이 다르므로)
    if np.any(np.isnan(X_tr)) or np.any(np.isnan(y_tr)) or np.any(np.isinf(X_tr)) or np.any(np.isinf(y_tr)):
        raise ValueError("Invalid values in training data")
    if np.any(np.isnan(X_val)) or np.any(np.isnan(y_val)) or np.any(np.isinf(X_val)) or np.any(np.isinf(y_val)):
        raise ValueError("Invalid values in validation data")
    
    return X_tr, y_tr, X_val, y_val


def calculate_metrics(y_true, y_pred):
    # 1차원으로 flatten
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    r2 = r2_score(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return {
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,        
    }

def train_and_evaluate():
    try:
        model = load_model()
        if model is None:
            print("ERROR: Failed to load model")
            print("R2: 0.000000")
            print("RMSE: inf")
            print("MAE: inf")
            print("MSE: inf")
            sys.stdout.flush()
            return
        compile_model(model)
        
        # 데이터 로드
        X_tr, y_tr, X_val, y_val = load_data_optimized()
        
        print(f"INFO: Data loaded - X_tr: {X_tr.shape}, y_tr: {y_tr.shape}, X_val: {X_val.shape}, y_val: {y_val.shape}")
        sys.stdout.flush()
        
        # 데이터셋 생성 시 shape 문제 해결
        try:
            full_train_dataset = tf.data.Dataset.from_tensor_slices((X_tr, y_tr))
            full_train_dataset = full_train_dataset.shuffle(buffer_size=len(X_tr), seed=42)
        except Exception as e:
            print(f"ERROR: Dataset creation failed: {e}")
            print(f"ERROR: X_tr dtype: {X_tr.dtype}, y_tr dtype: {y_tr.dtype}")
            print("R2: 0.000000")
            print("RMSE: inf")
            print("MAE: inf") 
            print("MSE: inf")
            sys.stdout.flush()
            return

        # 데이터셋 크기 계산 및 분할 지점 설정 (80% 훈련, 20% 검증)
        dataset_size = len(X_tr)
        val_size = int(0.2 * dataset_size)
        
        val_dataset = full_train_dataset.take(val_size)
        train_dataset = full_train_dataset.skip(val_size)

        train_dataset = train_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)  
            
        callbacks = []    
        # Early stopping
        early_stop_patience = 10 if mode == "cv" else 15
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=early_stop_patience,
            restore_best_weights=True, 
            verbose=0
        ))    
        # Learning rate reduction
        lr_patience = 5 if mode == "cv" else 5
        callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_patience,
            verbose=0
        ))    
        # Optuna trial callback (있는 경우)
        if trial_number is not None and mode == "test":
            class ReportCB(tf.keras.callbacks.Callback):
                def on_epoch_end(self, epoch, logs=None):
                    if logs and 'val_loss' in logs:
                        value = -logs['val_loss']
                        print(f"intermediate_value:{epoch}:{value}")
                        sys.stdout.flush()
            callbacks.append(ReportCB())
        
        try:
            history = model.fit(
                train_dataset,
                epochs=EPOCHS,
                validation_data=val_dataset,
                callbacks=callbacks,
                verbose=0,
            )
            
            # 예측 - 배치 예측으로 속도 향상
            print(f"INFO: Starting prediction on {len(X_val)} samples")
            sys.stdout.flush()
            
            y_pred = model.predict(X_val, batch_size=BATCHSIZE*2, verbose=0)
            
            # 예측값이 2D인 경우 1D로 변환
            if len(y_pred.shape) > 1:
                y_pred = y_pred.flatten()
            
            print(f"INFO: Prediction complete - shape: {y_pred.shape}")
            sys.stdout.flush()
                    
            # 예측값 검증
            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                print("ERROR: Invalid predictions (NaN or Inf)")
                print("R2: 0.000000")
                print("RMSE: inf")
                print("MAE: inf")
                print("MSE: inf")
                sys.stdout.flush()
                return
            
            # 메트릭 계산
            metrics = calculate_metrics(y_val, y_pred)
            
            # 결과 출력 - 정확한 형식으로
            print(f"R2: {metrics['r2']:.6f}")
            print(f"RMSE: {metrics['rmse']:.6f}")
            print(f"MAE: {metrics['mae']:.6f}")
            print(f"MSE: {metrics['mse']:.6f}")
            sys.stdout.flush()
            
            # 예측값을 npy 파일로 저장
            try:
                pred_path = "predictions.npy"
                np.save(pred_path, y_pred)
                print(f"INFO: Predictions saved to {pred_path}")
                
                # 파일이 제대로 저장되었는지 확인
                if os.path.exists(pred_path):
                    saved_pred = np.load(pred_path)
                    print(f"INFO: Verified predictions saved, shape: {saved_pred.shape}")
                else:
                    print(f"ERROR: Failed to save predictions to {pred_path}")
                sys.stdout.flush()
            except Exception as e:
                print(f"ERROR: Exception while saving predictions: {e}")
                sys.stdout.flush()
                
            # 추가 통계 정보 (디버깅용)
            if os.environ.get('DEBUG', '0') == '1':
                print(f"Best_epoch: {len(history.history['loss']) - early_stop_patience}")
                print(f"Final_val_loss: {min(history.history['val_loss']):.6f}")
                
        except Exception as e:
            import traceback
            print(f"ERROR: Exception during training: {e}")
            print(f"ERROR: Traceback: {traceback.format_exc()}")
            print("R2: 0.000000")
            print("RMSE: inf")
            print("MAE: inf")
            print("MSE: inf")
            sys.stdout.flush()
            
    except Exception as e:
        import traceback
        print(f"ERROR: Fatal error in train_and_evaluate: {e}")
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        print("R2: 0.000000")
        print("RMSE: inf")
        print("MAE: inf")
        print("MSE: inf")
        sys.stdout.flush()

def clear_memory():
    tf.keras.backend.clear_session()
    for _ in range(3):
        gc.collect()
        
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass
        
if __name__ == "__main__":
    try:
        print(f"INFO: Starting training with args: {sys.argv[1:]}")
        sys.stdout.flush()
        train_and_evaluate()
    except Exception as e:
        import traceback
        print(f"ERROR: Fatal error in learning process: {e}")
        print(f"ERROR: Traceback: {traceback.format_exc()}")
        print("R2: 0.000000")
        print("RMSE: inf")
        print("MAE: inf")
        print("MSE: inf")
        sys.stdout.flush()
    finally:
        clear_memory()