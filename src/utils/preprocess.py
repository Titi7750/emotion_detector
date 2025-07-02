import pandas as pd
import numpy as np

def load_and_preprocess(csv_path):
    ''' This function loads and preprocesses the dataset from a CSV file. '''

    emotion_labels = {
        0: "Colère",
        1: "Dégoût",
        2: "Peur",
        3: "Joie",
        4: "Tristesse",
        5: "Surprise",
        6: "Neutre"
    }

    def process_pixels(pixels):
        return np.array([int(pixel) for pixel in pixels.split()]).reshape(48, 48)

    # Read the CSV file into a DataFrame
    dataframe = pd.read_csv(csv_path)

    # Apply pixel processing
    dataframe['pixels'] = dataframe['pixels'].apply(process_pixels)
    
    '''
    Normalization of pixels
    The pixel values are initially between 0 and 255, we normalize them between 0 and 1
    to improve the convergence of the machine learning model
    '''
    dataframe['pixels'] = dataframe['pixels'] / 255.0

    train = dataframe[dataframe['Usage'] == 'Training']
    test = dataframe[dataframe['Usage'] == 'PublicTest']

    X_train = np.stack(train['pixels'].values).reshape(-1, 48, 48, 1)
    y_train = pd.get_dummies(train['emotion']).values

    X_test = np.stack(test['pixels'].values).reshape(-1, 48, 48, 1)
    y_test = pd.get_dummies(test['emotion']).values

    return X_train, y_train, X_test, y_test, emotion_labels
