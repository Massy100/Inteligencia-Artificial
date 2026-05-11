import cv2
import mediapipe as mp
import math
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def distancia(p1, p2):
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)

cap = cv2.VideoCapture(0)  # 0 = camara predeterminada

# Rangos de calibracion
DISTANCIA_MIN = 20    # Distancia mmin (dedos juntos)
DISTANCIA_MAX = 200   # Distancia max (dedos separados)

with mp_hands.Hands(
    static_image_mode=False,  
    max_num_hands=2,        
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("No se pudo capturar la imagen")
            continue
        
        # Voltear imagen para efecto espejo
        image = cv2.flip(image, 1)
        imagen_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Procesar la imagen para detectar manos
        resultados = hands.process(imagen_rgb)
        
        # Variables para almacenar el valor de la barra
        valor_barra = 0
        distancia_actual = 0
        color_linea = (0, 255, 0)  # Verde por defecto
        
        if resultados.multi_hand_landmarks:
            for hand_landmarks in resultados.multi_hand_landmarks:
                # Dibujar los landmarks y conexiones de la mano
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Obtener las coordenadas de los landmarks
                landmarks = hand_landmarks.landmark
                h, w, _ = image.shape
                
                # Landmark 4: Punta del pulgar
                # Landmark 8: Punta del índice
                pulgar_tip = landmarks[4]
                indice_tip = landmarks[8]
                
                # Calcular distancia entre pulgar e índice
                distancia_actual = distancia(pulgar_tip, indice_tip)
                
                # Mapear la distancia a un rango de 0 a 100 usando np.interp
                valor_barra = np.interp(
                    distancia_actual, 
                    [DISTANCIA_MIN, DISTANCIA_MAX], 
                    [0, 100]
                )
                
                # Limitar el valor entre 0 y 100
                valor_barra = max(0, min(100, valor_barra))
                
                # Cambiar color de la línea según la distancia
                if distancia_actual < DISTANCIA_MIN + 10:  # Dedos muy cerca
                    color_linea = (0, 0, 255)  # Rojo
                elif distancia_actual > DISTANCIA_MAX - 50:
                    color_linea = (0, 255, 0)   # Verde
                else:
                    color_linea = (0, 255, 255) # Amarillo
                
                # Convertir coordenadas normalizadas a píxeles para dibujar
                pulgar_px = (int(pulgar_tip.x * w), int(pulgar_tip.y * h))
                indice_px = (int(indice_tip.x * w), int(indice_tip.y * h))
                
                # Dibujar línea entre pulgar e índice
                cv2.line(image, pulgar_px, indice_px, color_linea, 3)
                
                # Dibujar círculos en los puntos de referencia
                cv2.circle(image, pulgar_px, 8, (255, 0, 0), -1)
                cv2.circle(image, indice_px, 8, (255, 0, 0), -1)
                
                # Mostrar distancia en tiempo real
                cv2.putText(image, f'Distancia: {distancia_actual:.1f}', (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Crear barra de estado horizontal
        barra_x = 50
        barra_y = image.shape[0] - 50
        barra_ancho = 300
        barra_alto = 30
        
        cv2.rectangle(image, (barra_x, barra_y), 
                     (barra_x + barra_ancho, barra_y + barra_alto), 
                     (100, 100, 100), -1)
        
        ancho_lleno = int(barra_ancho * (valor_barra / 100))
        
        # Cambiar color de la barra según el valor
        if valor_barra < 30:
            color_barra = (0, 0, 255)  # Rojo (valor bajo)
        elif valor_barra < 70:
            color_barra = (0, 255, 255)  # Amarillo (valor medio)
        else:
            color_barra = (0, 255, 0)  # Verde (valor alto)
        
        cv2.rectangle(image, (barra_x, barra_y), 
                     (barra_x + ancho_lleno, barra_y + barra_alto), 
                     color_barra, -1)
        
        cv2.rectangle(image, (barra_x, barra_y), 
                     (barra_x + barra_ancho, barra_y + barra_alto), 
                     (255, 255, 255), 2)
        
        # Texto con el valor porcentual
        cv2.putText(image, f'{int(valor_barra)}%', 
                    (barra_x + barra_ancho + 10, barra_y + barra_alto - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Texto de instrucciones
        cv2.putText(image, 'Air Controller - Mueve pulgar e indice', (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, 'Presiona Q para salir', (10, image.shape[0] - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow('Air Controller - Control con Manos', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()