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

while True:
	date = datetime.datetime.now(datetime.timezone.utc)
	# Call the clouds object to determine sky status
	result=clouds.classify()
	# Publish the result to MQTT
	client.publish(mqtt_topic, result)
	# Print the result to the console
	print(result)

	time.sleep(30)

