''' Python version: 3.11.9 '''

from src.utils.train import train

def main():
    ''' Main function to start training '''
    
    csv_path = "data/fer2013.csv"
    train(csv_path)

if __name__ == "__main__":
    main()
