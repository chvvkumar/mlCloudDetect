import sys
from pathlib import Path
import time
from datetime import datetime
from datetime import timedelta
import numpy as np
from PIL import Image
import logging
import os
import json
import requests
from io import BytesIO
import configparser

# import sqlite3
# import argparse
# import cv2
# import PIL

from mcpConfig import McpConfig
config=McpConfig()

logger = logging.getLogger("mcpClouds")

# Suppress Tensorflow warnings
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import keras
logger.setLevel(logging.INFO)

# Add the parent directory to the path so we can import the config
sys.path.append(str(Path(__file__).parent.absolute().parent))

class McpClouds:
    # Initialize the object
    def __init__(self):
        self.config = config
        logger.info('Using keras model: %s', config.get("KERASMODEL"))
        self.model = keras.models.load_model(config.get("KERASMODEL"), compile=False)
        # Read trainfolder from config and list directories
        trainfolder = config.get("TRAINFOLDER")
        self.CLASS_NAMES = self._get_class_names(trainfolder)
        logger.info('Class names: %s', self.CLASS_NAMES)

    # Get the class names from the training folder
    def _get_class_names(self, trainfolder):
        try:
            class_names = [d.name for d in Path(trainfolder).iterdir() if d.is_dir()]
            print(f"Class names: {class_names}")
            return class_names
        except Exception as e:
            logger.error('Error reading class names from trainfolder: %s', e)
            class_names = self.config.get("CLASS_NAMES").split(",")
            logger.info('Loaded class names from config file: %s', class_names)
            return class_names

    # Classify the image
    def classify(self):
        image_url = config.get("LATEST_FILE_URL")
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content))
        result = self.detect(image)
        return result

    # Detect the image
    def detect(self, image):
        # Load and preprocess the image
        image = image.resize((256, 256))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        detect_start = time.time()

        # Predicts the model
        prediction = self.model.predict(image_array, verbose=0)
        idx = np.argmax(prediction)
        class_name = self.CLASS_NAMES[idx]
        confidence_score = float(prediction[0][idx])  # Convert to native Python float
 
        # Calculate the elapsed time
        detect_elapsed_s = time.time() - detect_start
        logger.info('Cloud detection in %0.4f s', detect_elapsed_s)
        logger.info('Rating: %s, Confidence %0.3f', class_name, confidence_score)
        
        # Create a dictionary with the required values
        result = {
            "class_name": class_name,
            "confidence_score": confidence_score,
            "detect_elapsed": detect_elapsed_s
        }

        # Convert the dictionary to a JSON string and return it
        return json.dumps(result)