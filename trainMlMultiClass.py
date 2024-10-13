import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, MaxPooling2D
import os
import cv2
from PIL import Image
import numpy

# Set up logging
import logging
logFilename = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'mlCloudDetect.log')
logger = logging.getLogger()
fhandler = logging.FileHandler(filename=logFilename, mode='a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.INFO)

if os.name == 'nt':
    _ = os.system('cls')
else:
    _ = os.system('clear')
print("trainMlCloudDetect by Gord Tulloch gord.tulloch@gmail.com V1.0 2024/09/04")
print("Usage: trainMlCloudDetect with no parameters. See mlCloudDetect.ini for input parameters")

VERSION = '1.0'

logger.info("Program Start - trainMlCloudDetect" + VERSION)

from mcpConfig import McpConfig
config = McpConfig()

dataDir = config.get('TRAINFOLDER')

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
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')  # 10 classes
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Custom callback to log progress
class LoggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        logger.info(f"Starting epoch {epoch + 1}")

    def on_epoch_end(self, epoch, logs=None):
        logger.info(f"Finished epoch {epoch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")

    def on_batch_end(self, batch, logs=None):
        logger.info(f"Finished batch {batch + 1}, loss: {logs['loss']}, accuracy: {logs['accuracy']}")

# Train the model
model.fit(normalized_train_ds, validation_data=normalized_val_ds, epochs=10, callbacks=[LoggingCallback()])

# Evaluate the model
loss, accuracy = model.evaluate(normalized_val_ds)
logger.info(f"Validation Loss: {loss}")
logger.info(f"Validation Accuracy: {accuracy}")
