# Librerias necesarias para procesamiento y entrenamiento
import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd
import pickle
import pygame

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

mp_drawing = mp.solutions.drawing_utils   #Ayudas para dibujar
mp_holistic = mp.solutions.holistic       #Soluciones de Mediapipe

# Conectamos a la cámara frontal (Webcam)
cap = cv2.VideoCapture(0)

# Iniciamos el modelo holístico
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():

        # Se guarda la imagen en frame
        ret, frame = cap.read()      
        
        # Reformatear: BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Para proteger las imágenes
        image.flags.writeable = False
        
        #Aplicamos el process para obtener el modelo holistico:
        results = holistic.process(image)               
        
        # RGB -> BGR para renderizar
        image.flags.writeable = True  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )

        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        # Mostramos la imagen                
        cv2.imshow('Raw Webcam Feed', image)

        # Para salir, pulsamos la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

results.face_landmarks.landmark[0]ç

print("x:", results.pose_landmarks.landmark[0].x,"y:", 
      results.pose_landmarks.landmark[0].y,"z:", 
      results.pose_landmarks.landmark[0].z)

#Tenemos 501 coordenadas en total
num_coords = len(results.pose_landmarks.landmark)+len(results.face_landmarks.landmark)
num_coords

#la primera columna servirá de identificación de clase 
landmarks = ['class']

for val in range(1,num_coords+1):
    landmarks += ['x{}'.format(val), 'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]

#Mostramos ejemplo:
print(landmarks[0:13])

with open('coords.csv', mode='w', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(landmarks)

class_name = "Cansado"

#Conectamos a la cámara frontal (Webcam)
cap = cv2.VideoCapture(0)

# Iniciamos el modelo holístico
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():

        # Se guarda la imagen en frame
        ret, frame = cap.read()
        
        # #Reformatear: BGR -> RGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Para proteger las imágenes
        image.flags.writeable = False
        
        #Aplicamos el process para obtener el holistico
        results = holistic.process(image)

        # RGB -> BGR para renderizar
        image.flags.writeable = True  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
  
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        # Exportamos las coordenadas
        try:

            # Guardamos los datos del pose y face en un np array
            # Extraemos Pose landmarks ( el método .flatten() convierte el np array en 1D)
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                      for landmark in pose]).flatten())
           
            # Extraemos Face landmarks            
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                      for landmark in face]).flatten())
        
            row = pose_row + face_row
            row.insert(0,class_name)
            
            #Exportar a CSV
            with open('coords.csv',mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            pass
        
        #Mostramos la imagen
        cv2.imshow('Raw Webcam Feed', image)

        # Para salir, pulsamos la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

class_name = "Despierto"

#Conectamos a la cámara frontal (Webcam)
cap = cv2.VideoCapture(0)

# Iniciamos el modelo holístico
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():

        # Se guarda la imagen en frame
        ret, frame = cap.read()
        
        # #Reformatear: BGR -> RGR
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        #Para proteger las imágenes
        image.flags.writeable = False
        
        #Aplicamos el process para obtener el holistico
        results = holistic.process(image)

        # RGB -> BGR para renderizar
        image.flags.writeable = True  
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 1. Face landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                                 mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                                 mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                                 )
        
        # 2. Right hand
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )
  
        # 3. Left Hand
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        # Exportamos las coordenadas
        try:

            # Guardamos los datos del pose y face en un np array
            # Extraemos Pose landmarks ( el método .flatten() convierte el np array en 1D)
            pose = results.pose_landmarks.landmark
            pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                      for landmark in pose]).flatten())
           
            # Extraemos Face landmarks            
            face = results.face_landmarks.landmark
            face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility]
                                      for landmark in face]).flatten())
        
            row = pose_row + face_row
            row.insert(0,class_name)
            
            #Exportar a CSV
            with open('coords.csv',mode='a', newline='') as f:
                csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(row)

        except:
            pass
        
        #Mostramos la imagen
        cv2.imshow('Raw Webcam Feed', image)

        # Para salir, pulsamos la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

# Leemos el archivo .csv
df = pd.read_csv("coords.csv")
df

# Guardaremos las variables en X y el target en Y
X = df.drop('class',axis=1)
y = df['class']

# Dividimos el dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1000)

# Definimos los cuatro pipelines
pipelines = {
    'lr':make_pipeline(StandardScaler(), LogisticRegression()),
    'rc':make_pipeline(StandardScaler(), RidgeClassifier()),
    'rf':make_pipeline(StandardScaler(), RandomForestClassifier()),
    'gb':make_pipeline(StandardScaler(), GradientBoostingClassifier()),
}

# Guardamos los modelos en la biblioteca fit_models
fit_models = {}
for algo, pipeline in pipelines.items():
    model = pipeline.fit(X_train, y_train)
    fit_models[algo] = model

# Probamos a predecir X_test con el modelo de Random Forest
fit_models['rf'].predict(X_test)

#Predecir las clases de X_test
for algo, model in fit_models.items():
    yhat = model.predict(X_test)
    
    #Mostramos las precisiones de cada modelo
    print(algo, accuracy_score(y_test, yhat))

# Escribimos el modelo en un archivo .pkl
with open('cansado_despierto.pkl', 'wb') as f:
    pickle.dump(fit_models['rf'], f)

# Abrimos de nuevo el modelo
with open('cansado_despierto.pkl', 'rb') as f:
    model = pickle.load(f)

# Inicializar pygame mixer
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('alarma.wav')  # Asegúrate de que el archivo de sonido esté en formato .wav

# Conectamos la cámara
cap = cv2.VideoCapture(0)

# Inicializar la variable de control de la alarma
alarm_playing = False

# Inicia el modelo holístico
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():

        # Se guarda la imagen en frame
        ret, frame = cap.read()

        # Verificar que la imagen se haya leído correctamente
        if not ret:
            print("No se pudo recibir imagen de la cámara. Saliendo...")
            break

        # BGR -> RGB y seguridad de la imagen
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Detecciones
        results = holistic.process(image)

        # RGB -> BGR y seguridad de la imagen
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Dibujo de landmarks
        # 1. Face landmarks
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
            mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        )

        # 2. Right hand
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
        )

        # 3. Left Hand
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
        )

        # 4. Pose Detections
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )

        # Exporta coordenadas
        try:
            # Verificar que se hayan detectado landmarks de pose y rostro
            if results.pose_landmarks and results.face_landmarks:
                # Extraemos Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([
                    [landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose
                ]).flatten())

                # Extraemos Face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([
                    [landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face
                ]).flatten())

                # Concatenamos las filas
                row = pose_row + face_row

                # Hacemos detecciones
                X = pd.DataFrame([row])

                # Predecir la clase
                body_language_class = model.predict(X)[0]

                # Predecir la probabilidad de la clase
                body_language_prob = model.predict_proba(X)[0]

                # Calcular la probabilidad
                probability = round(body_language_prob[np.argmax(body_language_prob)], 2)

                # Mostrar valores para depuración
                print(f"Clase predicha: '{body_language_class}'")
                print(f"Probabilidad: {probability}")

                # Recogemos coordenadas de la oreja izquierda para visualizar texto dinámicamente
                coords = tuple(np.multiply(
                    np.array(
                        (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                         results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
                    ), [640,480]).astype(int))

                # Mostramos la clase en la oreja izquierda
                cv2.rectangle(
                    image,
                    (coords[0], coords[1]+5),
                    (coords[0]+len(body_language_class)*20, coords[1]-30),
                    (245, 117, 16), -1)
                cv2.putText(
                    image, body_language_class, coords,
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Status Box
                cv2.rectangle(image, (0,0), (250, 60), (245, 117, 16), -1)

                # Mostramos Class
                cv2.putText(
                    image, 'CLASS', (95,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    image, body_language_class.split(' ')[0], (90,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Mostramos Probabilidad
                cv2.putText(
                    image, 'PROB', (15,12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                cv2.putText(
                    image, str(probability), (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Al estar muy cansado, activamos la alerta de descanso y la alarma
                if (probability > 0.70) and ("Cansado" in body_language_class):

                    # Mostrar mensaje de alerta
                    cv2.rectangle(
                        image,
                        (coords[0], coords[1]+5),
                        (coords[0]+len('MUY CANSADO')*20, coords[1]-30),
                        (0, 0, 255), -1)

                    cv2.putText(
                        image, 'MUY CANSADO', coords,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    cv2.rectangle(image, (0,0), (550, 60), (0, 0, 255), -1)

                    cv2.putText(
                        image, "ALERTA! NECESITAS DESCANSAR", (15,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Activar alarma si no está sonando
                    if not alarm_playing:
                        alarm_playing = True
                        alarm_sound.play(-1)  # Reproducir en bucle
                else:
                    if alarm_playing:
                        alarm_playing = False
                        alarm_sound.stop()
            else:
                print("No se detectaron landmarks de pose o rostro.")
                # Puedes manejar este caso si es necesario

        except Exception as e:
            print(f"Ocurrió un error en el procesamiento: {e}")
            pass

        # Mostrar la imagen
        cv2.imshow('DETECTOR DE SOMNOLENCIA', image)

        # Salir con la tecla 'q'
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Liberar la cámara y cerrar ventanas
    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()