import os
import numpy as np
import cv2
import pickle
from PIL import Image

#Cascada para declarar zona de interes, Cord1 y Cord2
cascada_facial = cv2.CascadeClassifier('cascadas/data/haarcascade_frontalface_alt2.xml')
recon = cv2.face.LBPHFaceRecognizer_create()

#Declaracion de paths de lectura de imagenes
#Lectura de directorios utilizando {{OS}}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imagenes_dir = os.path.join(BASE_DIR, "imagenes") #Nombre de carpeta con imagenes para entramiento

id_actual = 0
dicc_etiquetas = {} #dicc: "Diccionario"
#Arrays class para NumPy ARRAY
y_etiquetas = []
x_entrenamiento = []

#zdi_vgray = gray[oy:oy+oh, ox:ox+ow]

#Leer todas las imagenes en carpeta 
#Conversion de imagen a NP array
for root, dirs, files in os.walk(imagenes_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            etiqueta = os.path.basename(root).replace(" ", "-").lower()
            #Crear diccionario de etiquetas y su destino 
            if not etiqueta in dicc_etiquetas:
                dicc_etiquetas[etiqueta] = id_actual
                id_actual += 1
            id_ = dicc_etiquetas[etiqueta]
            imagen_pil = Image.open(path).convert("L") #Cambiar a escala de grises
            size = (550, 550) #Re-Escala de imagenes en el 
            img_final = imagen_pil.resize(size, Image.ANTIALIAS)
            array_imagen = np.array(img_final, "uint8") #NumPy
            caras = cascada_facial.detectMultiScale(array_imagen, scaleFactor=1.5, minNeighbors=5)
            #ojos = cascada_visual.detectMultiScale(zdi_vgray)

            #Declarar zona de interes y enlazar con NP Array
            for (x, y, w, h) in caras:
                zdi = array_imagen[y:y+h, x:x+w]
                x_entrenamiento.append(zdi) #enlace
                y_etiquetas.append(id_)
#Mostrar stats
print(y_etiquetas, x_entrenamiento, array_imagen, etiqueta, path)
#print(y_etiquetas)
#print(x_entrenamiento)
#print(dicc_etiquetas,id_actual
#print(path)
#exportar etiquetas
#Usar Pickle para clasificar modulo facial
with open("DiccEtiquetas.pickle", 'wb') as f: #Escribir "DiccEtiquetas.pickle" como f(archivo)
    pickle.dump(dicc_etiquetas, f)

recon.train(x_entrenamiento, np.array(y_etiquetas))
recon.save("recon/entrenador-ronal.yml")