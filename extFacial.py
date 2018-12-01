import numpy as np
import cv2
import pickle

cascada_facial = cv2.CascadeClassifier('cascadas/data/haarcascade_frontalface_alt2.xml')

cap = cv2.VideoCapture(0)

while(True):
    #Capturar frame por frame
    ret, frame = cap.read()
    #Convertir BGR a escala de grises para inferencias. "cv2.COLOR_BGR2GRAY"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Valores de para clasificacion de acuerdo a la documentacion(OpenCV) y testing
    #"scaleFactor, minNeighbors"
    caras = cascada_facial.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #Para;  x/y h(-y)/w(-x) en modulo de det.Facial
    for (x, y, w, h) in caras:
        print(x,y,w,h)
        #Especificaciones para la region de interes(cara) de las lecturas:
        #zdi: "Zona de Interes"
        zdi_gray = gray[y:y+h, x:x+w]
        zdi_color = frame[y:y+h, x:x+w]
        #exportar zona de interes a un archivo PNG(10)
        img_item = "extRostros/rostro1.png"
        cv2.imwrite(img_item, zdi_color)
        img_item2 = "extRostros/rostro2.png"
        cv2.imwrite(img_item2, zdi_gray)
        img_item3 = "extRostros/rostro3.png"
        cv2.imwrite(img_item3, zdi_color)
        img_item4 = "extRostros/rostro4.png"
        cv2.imwrite(img_item4, zdi_gray)
        img_item5 = "extRostros/rostro5.png"
        cv2.imwrite(img_item5, zdi_color)
        img_item6 = "extRostros/rostro6.png"
        cv2.imwrite(img_item6, zdi_gray)
        img_item7 = "extRostros/rostro7.png"
        cv2.imwrite(img_item7, zdi_color)
        img_item8 = "extRostros/rostro8.png"
        cv2.imwrite(img_item8, zdi_gray)
        img_item9 = "extRostros/rostro9.png"
        cv2.imwrite(img_item9, zdi_color)
        img_item10 = "extRostros/rostro10.png"
        cv2.imwrite(img_item10, zdi_gray)
        
        #Rectangulo alrededor de la zona de interes(cara)
        #Declarar Ancho, Altura y borde para crear rectangulos de interpolacion
        color = (255,0,0) #La escala BGR es 0-255
        stroke = 2
        fin_cord_1 = x+w #Altura
        fin_cord_2 = y+h #Ancho
        #(frame, (x, y),(Ancho, Altura))
        cv2.rectangle(frame, (x, y),(fin_cord_1, fin_cord_2), color, stroke) #rectagulos(frame, (x, y) (x,w))

    #Mostrar el Frame resultante
    cv2.imshow('frame',frame)
    #Mostrar detect en color over escala de grises
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

#Cierre de captura al finalizar
cap.release()
cv2.destroyAllWindows()