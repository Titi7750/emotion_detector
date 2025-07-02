from src.utils.preprocess import load_and_preprocess
from src.models.cnn import first_cnn_model
import matplotlib.pyplot as plt

def train(csv_path, model_path="output/emotion_model.h5"):
    ''' Train the CNN model on the dataset from the CSV file. '''
    
    X_train, y_train, X_test, y_test, labels = load_and_preprocess(csv_path)

    model = first_cnn_model()
    history = model.fit(
        X_train,
        y_train,
        epochs=25,
        batch_size=64,
        validation_data=(
            X_test,
            y_test
        )
    )

    model.save(model_path)
    print(f"Modèle sauvegardé dans {model_path}")

    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='test')
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
