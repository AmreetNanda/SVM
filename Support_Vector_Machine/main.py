from data_utils import load_data
from preprocess import preprocess_split
from models import train

def run_training():
    print("Loading dataset...")
    df = load_data()

    print("Preprocessing dataset...")
    X, y, enc = preprocess_split(df)

    print("Training model...")
    train(X, y, enc)

if __name__ == "__main__":
    run_training()
