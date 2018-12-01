import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    #Capturar frame por frame
    ret, frame = cap.read()

    #Contador de FPS
    cv2.imshow('frame',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Cierre de captura al finalizar
cap.release()
cv2.destroyAllWindows()