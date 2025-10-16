''' Define a Convolutional Neural Network (CNN) model for emotion detection. '''

from keras.models import Sequential # Permet de créer un modèle couche par couche

# -----

from keras.layers import Conv2D # Regarde les petits morceaux d'une image pour repérer (bords, yeux, etc...)
from keras.layers import MaxPooling2D # Réduit la taille des images tout en gardant les informations importantes
from keras.layers import Dropout # Évite que le modèle mémorise trop les données d'entraînement
from keras.layers import Flatten # Transforme les données en liste de nombres
from keras.layers import Dense # Crée des couches de neurones pour apprendre des choses complexes

def first_cnn_model():
    ''' Build and compile a CNN model for emotion detection. '''

    model = Sequential([

        # Première couche de convolution
        # Nombre de filtres : 32
        # Taille des filtres : 3x3
        # Garde les valeurs importantes et ignore le reste (activation='relu')
        Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
        MaxPooling2D(2,2), # Réduit la taille de l'image de moitié
        Dropout(0.25), # Évite le surapprentissage

        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Dropout(0.25),

        # Transforme les données en liste de nombres
        Flatten(),

        # Nombre de neurones dans cette couche : 128
        # Garde les valeurs importantes et ignore le reste
        Dense(128, activation='relu'),
        Dropout(0.5), # Devient plus solide et évite de se tromper sur de nouvelles données

        # Nombre de neurones dans cette couche (7 émotions)
        # Donne une probabilité pour chaque émotion
        Dense(7, activation='softmax')
    ])

    model.compile(
        optimizer='adam', # Méthode pour apprendre rapidement et efficacement
        loss='categorical_crossentropy', # Mesure à quel point les prédictions sont justes ou fausses
        metrics=['accuracy'] # Pourcentage de bonnes réponses
    )

    return model
