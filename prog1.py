import cv2
import numpy as np
from time import sleep
from tensorflow.keras.models import load_model
from robodk.robolink import *
from robodk.robomath import *
from robodk.robodialogs import *


#----------------------------------------------
# Link to RoboDK
RDK = Robolink()

# Obtient le robot par son nom
robot = RDK.Item('Doosan Robotics M1013', ITEM_TYPE_ROBOT)  #


def draw_rectangle_on_piece(image, color):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Si aucune pièce n'est détectée, la fonction retourne False.
    if not contours:
        return False, None, None, None, None

    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    
    return True, x, y, w, h  # Indique qu'une pièce a été détectée.

# Charger le modèle entraîné
model = load_model(r'C:\Users\user\OneDrive - UQAR\Bureau\nouveaumodel\modeleVGG16.h5')
image_size = (150, 150)


# Get the camera
DEVICE_ID = 2
cap = cv2.VideoCapture(DEVICE_ID)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_resized = cv2.resize(frame, image_size)
    image_array = frame_resized / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_array)
    
    color = None
    
    detected, x, y, w, h = draw_rectangle_on_piece(frame, color)
    
    if prediction[0][0] > 0.5:
        result = 'avec_defauts'
        color = (0, 0, 255)  # Red
        detected, x, y, w, h = draw_rectangle_on_piece(frame, color)
        
    else:
        result = 'sans_defauts'
        color = (0, 255, 0)  # Green
        detected, x, y, w, h = draw_rectangle_on_piece(frame, color) 

              
    if detected:
        cv2.putText(frame, "Prediction: " + result, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    cv2.imshow('Video Feed', frame)

    key = cv2.waitKey(1)
    if key == ord('d'):
         cap.release()  # Arrêter la capture vidéo
         print("Arrêt de la capture vidéo. Exécution du programme robotique.")
         RDK.RunProgram("Prog2", True) 
         sleep(10)  # Attendre que le programme se termine, ajuster le temps d'attente selon le programme
         print("Reprise de la capture vidéo.")
         cap = cv2.VideoCapture(DEVICE_ID)  # Redémarrer la capture vidéo
         continue
        
    if key == ord('s'):
        path_to_save = f"C:/Users/user/Downloads/traitement de modele/assets/captured_image_{result}.jpg"
        cv2.imwrite(path_to_save, frame)
        print(f"Image enregistrée sous: {path_to_save}")

    
    
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
