import numpy as np
import random

# Dummy Tabular Model
def tabular_predict(input_data):
    # Simulate a regression model: sum inputs and add noise
    return np.sum(input_data) + random.uniform(-1, 1)

# Dummy Text Model
def text_predict(input_text):
    # Simulate sentiment analysis
    sentiments = ["Positive", "Neutral", "Negative"]
    return random.choice(sentiments)

# Dummy Image Model
def image_predict(image_data):
    # Simulate image classification
    classes = ["Cat", "Dog", "Car", "Tree"]
    return random.choice(classes)
