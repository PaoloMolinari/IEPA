import cv2 as cv
#import numpy as np
import time

from pythonosc import udp_client
#from pythonosc.osc_message_builder import OscMessageBuilder

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = mp.tasks.BaseOptions

GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
GestureRecognizerResult = mp.tasks.vision.GestureRecognizerResult

VisionRunningMode = mp.tasks.vision.RunningMode

# OSC Configuration
OSC_IP = "127.0.0.1"
OSC_PORT = 57120
osc_client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

frame_count = 0

def result(result: GestureRecognizerResult, output_image: mp.Image, timestamp_ms: int):
    #declarar variables globales de la funcion
    # para el lado de la mano, el id de la mano y el gesto
    hand_side = ""
    handID = ""
    gesture = ""

    # si el subcampo de los rsultados del reconocimeinto de gestos
    # es "handedness"
    if result and result.handedness:
        # para cada mano enumere los campos de handedness
        for hand_idx, handedness in enumerate(result.handedness):
            # creamos una lista para los datos
            # hand_data = []
            # para el handedness de cada mano
            for hand in handedness:
                # hand_data.extend([hand.index, hand.score, hand.display_name, hand.category_name])
                # guardamos en nuestras variables
                # el index de la mano y el lado de esta (derecha o izquierda)
                handID = hand.index
                hand_side = hand.category_name
                osc_client.send_message(f"/handedness/{hand_idx}", hand_side)
    # para el gesto de cada mano si pertenece al grupo "gestures"
    if result and result.gestures:
        # enumeramos las componentes de la lista "gestures"
        for hand_idx, gesturesResult in enumerate(result.gestures):
            # creamos
            #gesture_data = []
            # para cada mano copiamos en nuestra variable gesto
            # el gesto de cada mano
            for gest in gesturesResult:
                #gesture_data.extend([hand_side, gest.category_name])
                # copiamos la categoria de gesto en nuestra variable
                # gesture
                gesture = gest.category_name
                osc_client.send_message(f"/gesture/{hand_idx}", gesture)

    # ai el subitem es hand_landmarks
    if result and result.hand_landmarks:
        # creamos una lista llamada landmarks-data
        landmarks_data = []
        # creamos una lista que contendra todos
        # los datos que enviaremos por el mensaje OSC
        #all_data = []
        #all_data.extend([handID, hand_side, gesture])
        # para cada grupo de landmarks de cada mano
        for hand_idx, landmarkResult in enumerate(result.hand_landmarks):
            # para cada landmark de cada grupo de landmarks
            for i, land in enumerate(landmarkResult):
                # populamos el array o lista con el indice y las coordenadas
                # x, y, z de cada landmark.
                # landmarks_data.extend([i, land.x, land.y, land.z])
                landmarks_data.extend([land.x, land.y, land.z])
            # populamos el array "all_data" con el id de la mano
            # el lado de la mano, el gesto, y el array con las
            # coordendas de los landmarks
            #all_data.extend([handID, hand_side, gesture, landmarks_data])

            #all_data = all_data + landmarks_data
            #print(all_data)
            # enviamos el mensaje OSC
            osc_client.send_message(f"/landmarks/{hand_idx}", landmarks_data)
            #osc_client.send_message("/hand/", all_data)
            #all_data.clear()
            landmarks_data.clear()


# declara variable que guarda el camino al modelo
model_path = '/home/pemb/python/mediapipe/gesture_recognizer.task'

# opciones para el reconecedor de gestos
options = GestureRecognizerOptions(
    # lugar del modelo
    base_options=BaseOptions(model_asset_path=model_path),
    # el modo en que se hara correr el
    # reconecedor de gestos
    # este puede ser, STILL_IMAGE, VIDEO y LIVE_STREAM
    running_mode=VisionRunningMode.LIVE_STREAM,
    # cantidad de manos a capturar
    num_hands=2,
    # definicion de la funcion callback
    result_callback=result)

# inicializacion captura de video con openCV
cap = cv.VideoCapture(0)
# si el dispositivo de video no abre
if not cap.isOpened():
    # mensaje y salida
    print("Cannot open camera")
    exit()

# imprimir que se esta inicializando    
print("Starting hand tracking with LIVE_STREAM mode...")
# escribir en la consola el ip y puerto
# en el cual esta siendo enviado el mensaje OSC
print(f"Sending OSC messages to {OSC_IP}:{OSC_PORT}")

# Conenzar la instancia del reconocedor de gestos
with GestureRecognizer.create_from_options(options) as recognizer:
    # bucle infinito 
    while True:
        # captura cuadro por cuadro
        ret, frame = cap.read()
        flipimage = cv.flip(frame, 1)
        # si el cuadro no es recivio correctamente
        # es decir ret no es true
        # enviamos un mensaje y salimos del bucle
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # si ret es true, es decir recibimos un cuadro
        #convertimos la imagen del cuadro 
        # a un tipo de imagen mediaPipe SRGB
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=flipimage)

        #ingresamos la imagen como parametro
        # al metodo recognize_async
        recognizer.recognize_async(mp_image, time.time_ns() // 1_000_000)

        # mostramos la imagen de la camara
        cv.imshow('frame', flipimage)

        # si pulsamos la tecla "q" salimos del bucle
        if cv.waitKey(1) == ord('q'):
            break
#cerramos la captura
#y destrumimos todas las ventanas abiertas        
cap.release()
cv.destroyAllWindows()

    
