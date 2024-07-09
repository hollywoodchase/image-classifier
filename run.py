from src.train import train_model
from src.evaluate import evaluate_model
from src.visualize import plot_training_history

if __name__ == "__main__":
    history = train_model()
    evaluate_model()
    plot_training_history(history)
