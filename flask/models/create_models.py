import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Flatten

# Create and save the tabular model
def create_tabular_model():
    model = DummyRegressor(strategy="mean")
    model.fit([[1], [2], [3]], [10, 20, 30])  # Training on dummy data
    joblib.dump(model, "tabular_model.pkl")
    print("Tabular model saved as 'tabular_model.pkl'")

# Create and save the text model
def create_text_model():
    model = Sequential([
        Embedding(input_dim=1000, output_dim=16, input_length=10),
        GlobalAveragePooling1D(),
        Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy")
    model.fit(np.random.randint(1000, size=(10, 10)), np.random.randint(2, size=(10, 1)), epochs=1)
    model.save("text_model.h5")
    print("Text model saved as 'text_model.h5'")

# Create and save the image model
def create_image_model():
    model = Sequential([
        Flatten(input_shape=(224, 224, 3)),
        Dense(10, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    model.fit(np.random.random((10, 224, 224, 3)), np.random.randint(10, size=(10, 1)), epochs=1)
    model.save("image_model.h5")
    print("Image model saved as 'image_model.h5'")

if __name__ == "__main__":
    create_tabular_model()
    create_text_model()
    create_image_model()
