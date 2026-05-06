import cv2
import mediapipe as mp
import math

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

ruta = 'manos.jpg'
def distncia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
# captura de la imagen desde la webcam
cap = cv2.VideoCapture(0) # 0 = camara predeterminada

# Funcion para determinar si un dedo esta levantado
def dedo_levantado(landmarks, dedo_tip, dedo_dip, dedo_pip, dedo_mcp):
    return (landmarks.landmark[dedo_tip].y < landmarks.landmark[dedo_dip].y and
            landmarks.landmark[dedo_dip].y < landmarks.landmark[dedo_pip].y and
            landmarks.landmark[dedo_pip].y < landmarks.landmark[dedo_mcp].y)

with mp_hands.Hands(
    static_image_mode=False, # modo dinamico (mejor para video)
    max_num_hands=2, # numero maximo de manos a detectar
    min_detection_confidence=0.5,
    min_traking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo capturar la imagen")
            continue
        image = cv2.flip(image, 1)
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        resultados = hands.process(imagen_rgb)
        print('Handedness:', resultados.multi_handedness)
    
    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        
            # obtener las marcas como listas
            landmarks = hand_landmarks.landmark
            
            # definir los puntos de los dedos (segun mediapipe)
            pulgar_tip = 4
            indice_tip = 8
            medio_tip = 12
            anular_tip = 16
            peque_tip = 20
            
            # puntos para cada dedo (tip, dip, pip, mcp)
            dedos = {
                'pulgar': (pulgar_tip, 2, 1, 0),
                'indice': (indice_tip, 6, 5, 0),
                'medio': (medio_tip, 10, 9, 8),
                'anular': (anular_tip, 14, 13, 13),
                'peque': (peque_tip, 18, 17, 0)
            }
            
            # verificar que dedos estan levantados
            dedos_levantados = []
            for nombre, (tip, dip, pip, mcp) in dedos.items():
                if dedo_levantado(hand_landmarks, tip, dip, pip, mcp):
                    dedos_levantados.append(nombre)
            
            
            # mostrar los dedos levantados en la imagen
            cv2.putText(image, 'Dedos levantados: ' + ', '.join(dedos_levantados), (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            # mostrar imagen en tiempo real
            cv2.imshow('Manos', image)
            # salir con la tecla q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cv2.imshow('Imagen', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()