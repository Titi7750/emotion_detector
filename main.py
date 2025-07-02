''' Python version: 3.11.9 '''

from src.utils.train import train

if __name__ == "__main__":
    csv_path = "data/fer2013.csv"
    train(csv_path)
