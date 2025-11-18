import os
import pickle

BASE_DIR = os.path.dirname(__file__)

def predict(tip, sex, smoker, day, time, size):
    model_path = os.path.join(BASE_DIR, "model.pkl")
    encoder_path = os.path.join(BASE_DIR, "encoder.pkl")

    model = pickle.load(open(model_path, "rb"))
    encoders = pickle.load(open(encoder_path, "rb"))

    from .preprocess import preprocess_input
    X_input = preprocess_input(encoders, tip, sex, smoker, day, time, size)

    return model.predict(X_input)[0]
