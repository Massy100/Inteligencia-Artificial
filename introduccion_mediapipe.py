import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

ruta = 'manos.jpg'

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    imagen = cv2.imread(ruta)
    altura, ancho, _ = imagen.shape
    image = cv2.flip(imagen, 1)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    resultados = hands.process(imagen_rgb)
    print('Handedness:', resultados.multi_handedness)
    
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagen, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Manos', imagen)
    cv2.waitKey(0
)
    
def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

# Funcion para determinar si un dedo esta levantado
def dedo_levantado(landmarks, dedo_tip, dedo_dip, dedo_pip, dedo_mcp, umbral=0.1):
    return (landmarks[dedo_tip].y < landmarks[dedo_dip].y and
            landmarks[dedo_dip].y < landmarks[dedo_pip].y and
            landmarks[dedo_pip].y < landmarks[dedo_mcp].y)
    
with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    imagen = cv2.imread(ruta)
    altura, ancho, _ = imagen.shape
    image = cv2.flip(imagen, 1)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    resultados = hands.process(imagen_rgb)
    print('Handedness:', resultados.multi_handedness)
    
if resultados.multi_hand_landmarks:
    for hand_landmarks in resultados.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            imagen, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        