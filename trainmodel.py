
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications import VGG16
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import load_img, img_to_array

# Charger les données depuis un fichier CSV
data_path = r'C:\Users\user\OneDrive - UQAR\Bureau\nouveaumodel\modele.csv'  # Remplacez par le chemin de votre fichier
data = pd.read_csv(data_path)

# Chemin du dossier contenant les images
image_folder = r'C:\Users\user\OneDrive - UQAR\Bureau\nouveaumodel\iCloud Photos\iCloud Photos'  # Remplacez par le chemin de votre dossier d'images

# Prétraitement des images
image_size = (150, 150)  # Taille des images pour l'entrée du modèle
images = []
labels = []

for index, row in data.iterrows():
    image_path = os.path.join(image_folder, row['chemin_image'])
    if os.path.exists(image_path):
        image = load_img(image_path, target_size=image_size)
        image_array = img_to_array(image) / 255.0  # Normalisation
        images.append(image_array)
        labels.append(1 if row['etiquette_classe'] == 'avec_defauts' else 0)  # 0 pour 'sans_defauts', 1 pour 'avec_defauts'

images = np.array(images)
labels = np.array(labels)

# Séparation des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.3, random_state=42)

# Chargement de VGG16 pré-entraîné
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

# Gel des couches du modèle pré-entraîné pour éviter leur modification lors de l'entraînement
for layer in base_model.layers:
    layer.trainable = False

# Construction du modèle
model = Sequential([
    base_model,
    Flatten(),
    Dense(512, activation='relu',kernel_regularizer=l2(0.001)),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])


# Compilation du modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraînement du modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Évaluation du modèle
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Loss: {loss}, Accuracy: {accuracy}')

model_save_path = r'C:\Users\user\OneDrive - UQAR\Bureau\nouveaumodel\modeleVGG16.h5'
model.save(model_save_path)
print("Modèle enregistré avec succès.")
