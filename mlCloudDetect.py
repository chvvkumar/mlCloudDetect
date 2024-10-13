#!/usr/bin/env python3
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import time
from pysolar.solar import *
import datetime
import os
import requests
import paho.mqtt.client as mqtt
import json

import warnings
warnings.filterwarnings("ignore")

VERSION="1.0.0"

from mcpClouds import McpClouds
clouds=McpClouds()
from mcpConfig import McpConfig
config=McpConfig()

# MQTT configuration
mqtt_broker = "192.168.1.250"
mqtt_port = 1883
mqtt_topic = "Astro/CloudStatus"

# Initialize MQTT client
client = mqtt.Client()
client.connect(mqtt_broker, mqtt_port, 60)

#latestFile=config.get("ALLSKYFILE")

# # URL to fetch the latest file
# latest_file_url = config.get("LATEST_FILE_URL")

# # Fetch the latest file from the URL
# response = requests.get(latest_file_url)
# if response.status_code == 200:
#     latestFile = response.text.strip()
# else:
#     raise Exception(f"Failed to fetch latest file from URL: {latest_file_url}")


while True:
	# If the sun is up don't bother
	date = datetime.datetime.now(datetime.timezone.utc)

	# Call the clouds object to determine if it's cloudy
	result=clouds.classify()
	client.publish(mqtt_topic, result)
	print(result)

	time.sleep(30)

