from datetime import datetime
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
import numpy as np
import os
import sys
import ast
import re
import h5py
import pandas as pd


def find_best_checkpoint(JobID):
    log = pd.read_csv(f"/fred/oz149/Tyler/substructure_classifier_v2/ModelEval/TrainingLogs/{JobID}/trainingLog_{PSNR}.log")
    min_val_loss = log['val_loss'].min()
    checkpoint_idx = log[log['val_loss']==min_val_loss].index[0] + 1

    if checkpoint_idx < 10:
        checkpoint_idx = f"0{checkpoint_idx}"

    return checkpoint_idx

def hdf5_generator(dataset_path, batch_size, idx_array, Test=False):

    # Check that batch size is even
    if batch_size % 2 != 0:
        print("Batch size must be even. Exiting.")
        sys.exit()
    else:
        batch_size = batch_size // 2

    with h5py.File(dataset_path, "r") as f:

        total_samples = sum([len(idx) for idx in idx_array])

        for start_idx in range(0, total_samples//2, batch_size):

            end_idx = min(start_idx + batch_size, total_samples)
            
            perturbed_dataset = f['CONFIGURATION_1_images_preprocessed'][idx_array[1][start_idx:end_idx]]
            unperturbed_dataset = f['CONFIGURATION_2_images_preprocessed'][idx_array[0][start_idx:end_idx]]

            dataset = np.concatenate((unperturbed_dataset, perturbed_dataset), axis=0)
            labels = np.concatenate((np.zeros(unperturbed_dataset.shape[0]), np.ones(perturbed_dataset.shape[0])), axis=0)

            # Shuffle the dataset
            shuffle_idx = np.random.permutation(dataset.shape[0])

            dataset = dataset[shuffle_idx]
            labels = labels[shuffle_idx]

            yield (dataset, labels)


STARTTIME = datetime.now()
DATASET_PATH = str(sys.argv[1])
LR = float(sys.argv[2])
BATCH_SIZE = int(sys.argv[3])
OUT_DIR = str(sys.argv[4])
MODEL = str(sys.argv[5])

# Get test/validation indices
unperturbed_metadata = pd.read_csv(f'data/CONFIGURATION_2_metadata_{MODEL}.csv')
perturbed_metadata = pd.read_csv(f'data/CONFIGURATION_1_metadata_{MODEL}.csv')

perturbed_training_idx = perturbed_metadata[perturbed_metadata['TRAIN']==1].index.values
perturbed_validation_idx = perturbed_metadata[perturbed_metadata['TRAIN']==0].index.values

unperturbed_training_idx = unperturbed_metadata[unperturbed_metadata['TRAIN']==1].index.values
unperturbed_validation_idx = unperturbed_metadata[unperturbed_metadata['TRAIN']==0].index.values

training_idx_array = np.array([unperturbed_training_idx, perturbed_training_idx])
validation_idx_array = np.array([unperturbed_validation_idx, perturbed_validation_idx])

# Run checks
print("Training idx shape: ", training_idx_array[0].shape, training_idx_array[1].shape)
print("Validation idx shape: ", validation_idx_array[0].shape, validation_idx_array[1].shape)

# Get image shape
with h5py.File(f"{DATASET_PATH}dataset_preprocessed_{MODEL}.h5", 'r') as f:
    try:
        image_shape = f['CONFIGURATION_1_images_preprocessed'].shape[1::]
    except KeyError:
        print("KeyError: 'CONFIGURATION_1_images_preprocessed' not found in dataset. Ensure you are using the correct dataset.")
    f.close()

# Initialize the ResNet50 model
# model = ResNet50(
#         include_top=True,
#         weights=None,
#         input_tensor=None,
#         input_shape=image_shape,
#         pooling=None,
#         classes=2,
#         classifier_activation='softmax'
#         )

model = tf.keras.models.load_model(f"models/model_pretrained_sis.keras")

# Compile the model
model.compile(optimizer=Adam(learning_rate=LR), loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print(model.summary())

# Make it a dataset
train_dataset = tf.data.Dataset.from_generator(
    lambda: hdf5_generator(f"{DATASET_PATH}dataset_preprocessed_{MODEL}.h5", BATCH_SIZE, training_idx_array, Test=False),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, image_shape[0], image_shape[1], image_shape[2]], [None])
)

validation_dataset = tf.data.Dataset.from_generator(
    lambda: hdf5_generator(f"{DATASET_PATH}dataset_preprocessed_{MODEL}.h5", BATCH_SIZE, validation_idx_array, Test=True),
    output_types=(tf.float32, tf.float32),
    output_shapes=([None, image_shape[0], image_shape[1], image_shape[2]], [None])
)

# Prefetch the data for performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Define callbacks
csv_logger = CSVLogger(f'{OUT_DIR}/trainingLog.log', append=True)

model_checkpoint_callback = ModelCheckpoint(
    filepath=OUT_DIR + "/checkpoint_{epoch:02d}.keras",
    save_weights_only=False,
    monitor='val_accuracy',
    save_freq='epoch'
)

# Define EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor='val_loss',  # Monitoring validation loss
    patience=10,         # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # To log when training stops
    restore_best_weights=True  # This restores model weights from the epoch with the best value of the monitored metric
)

# Setup TensorBoard callback
tensorboard_callback = TensorBoard(
    log_dir=f"{OUT_DIR}/logs",  # Specify the path to save the logs
    histogram_freq=1,  # Frequency (in epochs) at which to compute activation and weight histograms for the layers of the model. If set to 0, histograms won't be computed.
    write_graph=True,  # Enables visualization of the graph in TensorBoard
    update_freq='epoch'  # Updates the logs after each epoch
)

model.fit(
    train_dataset,
    epochs=50,
    validation_data=validation_dataset,
    callbacks=[csv_logger, model_checkpoint_callback, early_stopping_callback, tensorboard_callback],  # Add the EarlyStopping callback here
    verbose=2
)

ENDTIME = datetime.now()
print(f"Training complete. Time taken: {ENDTIME - STARTTIME}")
print("Script Complete")