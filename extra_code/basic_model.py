import os
import gc
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.model_selection import train_test_split
import logging

# Environment settings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow INFO and WARNING messages
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_enable_xla_devices'
os.environ['XLA_FLAGS'] = '--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_gpu_force_compilation_parallelism=1'

# Suppress TensorFlow logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

BATCHSIZE = int(sys.argv[1])
EPOCHS = int(sys.argv[2])
fps_file = sys.argv[3]
y_true_file = sys.argv[4]

def load_model():
    with open('save_model/model_config.json', 'r') as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights('save_model/model_weights.weights.h5')
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.MeanSquaredError(), 
              metrics=[tf.keras.metrics.MeanSquaredError(), 
                       tf.keras.metrics.MeanAbsoluteError(), 
                       tf.keras.metrics.RootMeanSquaredError()])
    return model

def preprocess_data(xtr, ytr):
    buffer_size = min(10000, len(xtr))
    dataset = tf.data.Dataset.from_tensor_slices((xtr, ytr))
    dataset = dataset.shuffle(buffer_size=buffer_size).batch(BATCHSIZE).prefetch(tf.data.AUTOTUNE)
    return dataset

def train_model(model, train_dataset, epochs):
    model.fit(train_dataset, epochs=epochs, verbose=0)
    model.save('save_model/trained_model.keras')    
    return model

def clear_gpu_memory():
    tf.keras.backend.clear_session()
    gc.collect()
    print("GPU memory cleared.", file=sys.stderr)

if __name__ == "__main__":
    fps = np.load(fps_file)
    y_true = np.load(y_true_file)

    xtr, xte, ytr, yte = train_test_split(fps, y_true, test_size=0.2, random_state=42)
    train_dataset = preprocess_data(xtr, ytr)

    model = load_model()
    trained_model = train_model(model, train_dataset, EPOCHS)

    clear_gpu_memory()
