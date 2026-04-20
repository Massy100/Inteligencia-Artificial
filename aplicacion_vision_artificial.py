import cv2
import pytesseract
import matplotlib.pyplot as plt 

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

image = cv2.imread('auto.jpg')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.show()

blur = cv2.blur(gray, (3,3))
plt.imshow(blur, cmap='gray')
plt.show()

canny = cv2.Canny(blur,130,200)
canny = cv2.dilate(canny, None, iterations=1)
plt.imshow(canny, cmap='gray')
plt.show()

#Deteccion de contornos con OpenCV
contorno, _ = cv2.findContours(canny, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
print(contorno)

for c in contorno:
    area = cv2.contourArea(c)
    x, y, w, h = cv2.boundingRect(c)
    epsilon = 0.09*cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, epsilon, True)

    if len(approx) == 4 and area > 9000:
        print(f'Area:{area}')
        print(f'x:{x}; y:{y}; w:{w}; h:{h};')
        relAspect = float(w)/h
        print('Relacion de aspecto:', relAspect)
        cv2.drawContours(image, [approx], 0, (0, 255, 0), 3)
        if relAspect > 3:
            placa = gray[y:y+h, x:x+w]
            texto = pytesseract.image_to_string(placa, config='--psm 11'.strip())
            print(f'Placa: {texto}')
            cv2.rectangle(image, (x,y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(image, texto, (x-20, y-10), 1, 2.2, (0,255,0), 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
