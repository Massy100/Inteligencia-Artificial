import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

ruta = 'manos.jpg'

with mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5) as hands:

    imagen = cv2.imread(ruta)
    imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)

    resultados = hands.process(imagen_rgb)

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                imagen, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow('Manos', imagen)
    cv2.waitKey(0
)