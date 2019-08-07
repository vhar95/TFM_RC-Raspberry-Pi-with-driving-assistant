#!/usr/bin/python
#Importar Librerias (Camara,Opencv,Math,Tensorflow-Keras)
import picamera.array
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
import csv
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from skimage import color, exposure, transform
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical

#Funcion calcular distancia
def distancia (ancho, focal, perAncho):
	return (ancho*focal)/perAncho

#Funcion encontrar senal
def detectar(img):
	gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gris = cv2.GaussianBlur(gris,(5,5),0)
	canny = cv2.Canny(gris,35,125)
	_,contorno,hi =  cv2.findContours(canny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	c = max(contorno,key = cv2.contourArea)
	return cv2.minAreaRect(c)

#Cargar modelo keras
def cargar():
	json_file = open('/home/pi/kmodels.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("/home/pi/kmodels.h5")
	return loaded_model

#Cargar etiquetas senal
def etiqueta():
	with open('/home/pi/signnames.csv','r')as f:
    		reader = csv.reader(f)
    		your_list = list(reader)
	y_labels = list(map(lambda x: x[1], your_list[1:]))
	return y_labels

#Procesar imagen prediccion
def prep(array):
	tp = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
	tpr = cv2.resize(tp, (48,48))
	tpr = np.rollaxis(tpr,-1)
	tpn = tpr[np.newaxis,...]
    	return tpn

#Inicializar variables y cargar modelo
pesos = '/home/pi/kmodels.h5'
modelo = '/home/pi/kmodels.json'
lista = etiqueta()
loaded_model = cargar()
sd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
loaded_model.compile(loss="categorical_crossentropy", optimizer=sd, metrics=["accuracy"]) 

#Inicializar Camara
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = picamera.array.PiRGBArray(camera)

#Cargar modelo cascade 
classifier = cv2.CascadeClassifier('stop_sign.xml')
time.sleep(0.1)

#Variables 
know_D = 5.11811
know_W = 1.81165
#focal_length = 2228.57578287
focal_length = 2592*3.6/3.67

#Captura Frames Stream Camara
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
	image1 = frame.array
	image = np.copy(image1)
	tiempo_in_cas = time.time()
	
	#Detectamos objeto con el modelo cascade
	cascade_obj = classifier.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5,minSize=(20, 20),flags=cv2.CASCADE_SCALE_IMAGE)
	tiempo_fin_cas = time.time()
	tiempo_to_cas = tiempo_fin_cas - tiempo_in_cas
	
	#Si hay objeto, lo encontramos
	if (len(cascade_obj)):
    		cas = np.array([cascade_obj[0]])
    		stop_detected = True
    		for (x_pos, y_pos, width, height) in cas:

			#Dibujar rectangulo objeto detectado
		        cv2.rectangle(image, (x_pos+(-1), y_pos+(-1)), (x_pos+width-(-1), y_pos+height-(-1)), (255, 255, 255), 2)
       			crop1 = image[y_pos+(-1):y_pos+height-(-1),x_pos+(-1):x_pos+width-(-1)]
			crop = np.copy(crop1)
		        
			#Encontrar objeto y calcular distancia
			marker = detectar(crop1)
			dist = (distancia(know_W,focal_length,width)/12)*2.54

			#Detectar usando modelo keras
			tiempo_in_ke = time.time()
			max_index = np.argmax(loaded_model.predict(prep(crop)))
			tiempo_fin_ke = time.time()
			tiempo_to_ke = tiempo_fin_ke - tiempo_in_ke

			#Escribir resultados imagen
			cv2.putText(image, 'STOP('+str(round(dist,2))+' cm, ' + str(round(tiempo_to_cas,2))+' s)', (x_pos-10, y_pos+80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
			cv2.putText(image, '(Keras, ' + str(round(tiempo_to_ke,2))+' s)', (0+crop.shape[0]+3, 0+crop.shape[1]-40), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 255), 2)
	else:
    		stop_detected = False
		crop = np.zeros((100,100,3),np.uint8)


	filas,column,ch = crop.shape
	roi = image[0:filas, 0:column ]
	image[0:filas, 0:column] = crop
	cv2.imshow("Reconocimiento Senales",image)
	del(image)
	del(crop)
	key = cv2.waitKey(1) & 0xFF
	rawCapture.truncate(0)
	if key == ord("q"):
		break
       