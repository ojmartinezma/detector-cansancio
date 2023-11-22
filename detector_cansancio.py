from scipy.spatial import distance
from imutils import face_utils
import numpy as np
from pygame import mixer
import time
import dlib
import cv2

#Se inicializa el mixer que reproduce el sonido de alerta
mixer.init()
sound = mixer.Sound("alert.wav")

#Se establece el limite para detectar ojo cerrado
EYE_ASPECT_RATIO_THRESHOLD = 0.3

#Se establecen los frames seguidos que inician la alerta
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50
ALERT_COUNTER_TRESHOLD = 5

#Se inicializa el contador de frames con ojo cerrado
COUNTER = 0
ALERT_COUNTER = 0

#Se carga el detector de caras usando el modelo haar cascade
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

#Definimos la función que calcula el aspecto del ojo
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    eye_aspect_radio = (A+B) / (2*C)
    return eye_aspect_radio

#Se carga el detector de cara, junto con el conjunto de datos de predicción
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

#Se extraen las coordenadas de los ojos
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

#Se inicia la captura de la webcam
video_capture = cv2.VideoCapture(0)

#Se esperan 2 segundos para que la camara inicie
time.sleep(2)

while(True):
    #Se lee un frame, y se cambia a escala de grises
    ret, frame = video_capture.read()
    frame = cv2.flip(frame,1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Se pasa el frame al detector de caras
    faces = detector(gray, 0)

    #Se obtiene la ubicación de la cara
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Dibujamos un rectangulo azul alrededor de la cara
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)

    #Si se detecta una cara se ejecuta la detección de ojos
    for face in faces:

        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        #Se obtiene la ubicación de los ojos
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        #Se calcula el aspecto de cada ojo
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        #se promedia el aspecto de ambos ojos
        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        #Se dibuja el contorno de los ojos detectados
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        #Se verifica que no se haya alcanzado la alarma de cansancio acumulado
        if ALERT_COUNTER>=ALERT_COUNTER_TRESHOLD:
            if not mixer.get_busy():
                    sound.play()
            cv2.putText(frame, "************************ALERTA!*************************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "****************PARE Y TOME UN DESCANSO!******************", (10, 250),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "************CANSANCIO ACUMULADO DETECTADO!****************", (10,470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        #Si el aspecto es menor que el parámetro
        elif eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            #Si se supera el parámetro de ojos cerrados consecutivamente se ejecuta la alerta
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                if not mixer.get_busy():
                    sound.play()
                cv2.putText(frame, "************************ALERTA!*************************", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "*****************CANSANCIO DETECTADO!*******************", (10,470),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            sound.stop()
            if COUNTER>=EYE_ASPECT_RATIO_CONSEC_FRAMES:
                ALERT_COUNTER += 1
            COUNTER = 0

    #Se inicializa la ventana de vídeo y las teclas para salir
    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

video_capture.release()
cv2.destroyAllWindows()
