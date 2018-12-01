from imutils.video import FileVideoStream
from imutils.video import FPS
import numpy as np
import cv2
import pickle
#Interpolaciones con Cascada
cascada_facial = cv2.CascadeClassifier('cascadas/data/haarcascade_frontalface_alt2.xml')
cascada_visual = cv2.CascadeClassifier('cascadas/data/haarcascade_eye.xml')
cascada_sonrisa = cv2.CascadeClassifier('casacadas/data/haarcascade_smile.xml')

#Indentificador entrenado entrada/lectura
recon = cv2.face.LBPHFaceRecognizer_create()
recon.read("recon/entrenador-ronal.yml")

#importar diccionario de etiquetas e interpretar con clasificador
etiquetas = {"person_name": 1}
with open("DiccEtiquetas.pickle", 'rb') as f: #Escribir "DiccEtiquetas.pickle" como f(archivo)
    og_etiquetas = pickle.load(f)
    etiquetas = {v:k for k,v in og_etiquetas.items()} 

cap = cv2.VideoCapture(0)

while(True):
    
    #Capturar frame por frame
    ret, frame = cap.read()
    
    #NATIVO;Mostrar los frmaes por segundo:
    
    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
    if int(major_ver)  < 3 :
        fps = cap.get(cv2.cv.CV_CAP_PROP_FPS)
        print "Frames por segundo: {0}".format(fps)
    else :
        fps = cap.get(cv2.CAP_PROP_FPS)
        print "Frames por segundo : {0}".format(fps)

# construct the argument parse and parse the arguments
#ap = argparse.ArgumentParser(cap)
#ap.add_argument("-v", "--video", required=True)
#help = ("path to input video file")
#args = vars(ap.parse_args(cap))
 
    # start the file video stream thread and allow the buffer to
    # start to fill
    #print("[INFO] starting video file thread...")
    #fvs = FileVideoStream(args["video"]).start()
    #time.sleep(1.0)
 
    # start the FPS timer
    #fps = FPS().start()
# loop over frames from the video file stream
#while fvs.more():
    # grab the frame from the threaded video file stream, resize
    # it, and convert it to grayscale (while still retaining 3
    # channels)
    #frame = fvs.read()
    #frame = imutils.resize(frame, width=450)
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #frame = np.dstack([frame, frame, frame])
 
    # display the size of the queue on the frame
    #cv2.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
        #(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)    
 
    # show the frame and update the FPS counter
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
    #fps.update()

    #Convertir BGR a escala de grises para inferencias. "cv2.COLOR_BGR2GRAY"
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #Valores de para clasificacion de acuerdo a la documentacion(OpenCV) y testing
    #"scaleFactor, minNeighbors"
    caras = cascada_facial.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #Para;  x/y h(-y)/w(-x) en modulo de det.Facial
    for (x, y, w, h) in caras:
        #print(x,y,w,h)
        #Especificaciones para la zona de interes(cara) de las lecturas:
        #"rdi:zona de interes"
        zdi_gray = gray[y:y+h, x:x+w] #(CordY + Altura, CordX + Altura)
        zdi_color = frame[y:y+h, x:x+w]
        
        #Clasificador entrenado:
        id_, conf = recon.predict(zdi_gray) #Nivel de confianza(conf) y etiquetas(id_)
        #Niveles de confianza aceptables
        if conf>=40: #and conf <= 85:
            print(id_)
            print(etiquetas[id_])
            
            #Etiqueta con nombre
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = etiquetas[id_]
            color = (255,255,255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        #exportar zona de interes a un archivo PNG
        img_item = "rostro1.png"
        cv2.imwrite(img_item, zdi_color)

        #Rectangulo alrededor de la zona de interes(cara)
        #Declarar Ancho, Altura y borde para crear rectangulos de interpolacion
        color = (255,0,0) #La escala BGR es 0-255
        stroke = 2
        fin_cord_1 = x+w #Altura
        fin_cord_2 = y+h #Ancho
        #(frame, (x, y),(Ancho, Altura))
        cv2.rectangle(frame, (x, y),(fin_cord_1, fin_cord_2), color, stroke) #rectagulos(frame, (x, y) (x,w))
        
        #Reacuadros sobre Ojos
        ojos = cascada_visual.detectMultiScale(zdi_gray)
        for (ox,oy,ow,oh) in ojos:
            cv2.rectangle(zdi_color,(ox,oy),(ox+ow,oy+oh),(0,255,0),2)
        img_ojos = "ojos.png"
        cv2.imwrite(img_ojos, zdi_color)

        #Recuadro, sonrisa
        #sonrisa = cascada_sonrisa.detectMultiScale(zdi_color)
        #for (sx,sy,sw,sh) in sonrisa:
            #cv2.rectangle(zdi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),3)

    #Mostrar el Frame resultante
    cv2.imshow('frame',frame)
    #Mostrar detect en color over escala de grises
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break 
#Cierre de captura al finalizar
cap.release()
cv2.destroyAllWindows()