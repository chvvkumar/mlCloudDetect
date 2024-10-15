import tensorflow as tf
from tensorflow.keras import Sequential, models    
from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, MaxPooling2D
import os
import cv2
from PIL import Image
import numpy
import time
import configparser

# Set up logging
import logging
logFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mlCloudDetect.log')
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=logFilename, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

VERSION = '1.0'

# Train the model for however many epochs
EPOCHS = 50

logger.info("Program Start - trainMlCloudDetect" + VERSION)

from mcpConfig import McpConfig
config = McpConfig()

# Load the data directory from the config file
dataDir = config.get('TRAINFOLDER')
print(f"Data directory: {dataDir}")

# List all directories (classes) under the training folder
dir_count = [d for d in os.listdir(dataDir) if os.path.isdir(os.path.join(dataDir, d))]

# Count the number of directories
NUM_CLASSES = len(dir_count)

print(f"Number of classes found under {dataDir}: {NUM_CLASSES}")
logger.info(f"Number of classes: {NUM_CLASSES}")

# Load the data
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataDir,
    validation_split=0.2,
    subset="training",

    seed=123,
    image_size=(256, 256),
    batch_size=32
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataDir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(256, 256),
    batch_size=32
)

# Normalize the data
normalization_layer = tf.keras.layers.Rescaling(1./255)
normalized_train_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
normalized_val_ds = validation_dataset.map(lambda x, y: (normalization_layer(x), y))

# Define the model
model = Sequential([
#    Input(shape=(256, 256, 3)),
#    Conv2D(32, (3, 3), activation='relu'),
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(NUM_CLASSES, activation='softmax')  # 10 classes
])

# Custom callback to log progress
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()
        logger.info("Training started")

    def on_train_end(self, logs=None):
        train_duration = time.time() - self.train_start_time
        logger.info(f"Training finished in {train_duration:.2f} seconds")

    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Finished epoch {epoch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")

    def on_batch_end(self, batch, logs=None):
        logger.info(f"Finished batch {batch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")


# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


# Model checkpoint callback
checkpoint_filepath = "best.weights.keras"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=checkpoint_filepath,
                        save_weights_only=False,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)

# Early stopping callback
EarlyStopCallback = tf.keras.callbacks.EarlyStopping(
                        monitor='accuracy',
                        min_delta=0,
                        patience=5,
                        verbose=1,
                        restore_best_weights=True
                        )

# Load the model if it exists
if os.path.exists(checkpoint_filepath):
    model = tf.keras.models.load_model(checkpoint_filepath)
else:
    logger.warning(f"Checkpoint file {checkpoint_filepath} not found. Skipping model loading.")

# Callbacks
callbacks_list = [EarlyStopCallback, 
                  model_checkpoint_callback, 
                  LoggingCallback()]

# Train the model
history = model.fit(normalized_train_ds, 
                    validation_data=normalized_val_ds, 
                    epochs=EPOCHS, 
                    callbacks=callbacks_list)

# Evaluate the model
loss, accuracy = model.evaluate(normalized_val_ds)
logger.info(f"Validation Loss: {loss}")
logger.info(f"Validation Accuracy: {accuracy}")

# Save the model
model.save('mlCloudDetect.keras')
model.summary()
