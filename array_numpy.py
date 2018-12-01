import os
import numpy as np
import cv2
import pickle
from PIL import Image

#Declaracion de paths de lectura de imagenes
#Lectura de directorios utilizando {{OS}}
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imagenes_dir = os.path.join(BASE_DIR, "imagenes") #Nombre de carpeta con imagenes para entramiento

#Arrays class para NumPy ARRAY
y_etiquetas = []
x_entrenamiento = []

#Leer todas las imagenes en carpeta 
for root, dirs, files in os.walk(imagenes_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root, file)
            label = os.path.basename(root).replace(" ", "-").lower()
            imagen_pil = Image.open(path).convert("L") #ESCALA DE GRISES
            array_imagen = np.array(imagen_pil, "uint8")
with open("array/nparray.pickle", 'wb') as f: #Escribir "DiccEtiquetas.pickle" como f(archivo)
    pickle.dump(array_imagen, f)