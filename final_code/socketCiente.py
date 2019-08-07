#!/usr/bin/python
#Importar Librerias
import cv2
import io
import socket
import struct
import time
import numpy as np
import pickle
from matplotlib.pyplot import imshow
import zlib
import csv
import picamera.array
from picamera import PiCamera
import pigpio
import keras.backend.tensorflow_backend
from keras.backend import clear_session
from keras.models import load_model
from keras.models import model_from_json

from keras.optimizers import Adam
import tensorflow as tf
import threading
import os

#Conexion Socket Servidor
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect(('192.168.2.149', 8880))
connection = client_socket.makefile('wb')

#Inicializar Camera

camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = picamera.array.PiRGBArray(camera)
img_counter = 0
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]

#Inicializar Pin 6 Servo
pi = pigpio.pi()
pi.set_mode(6,pigpio.OUTPUT)
pi.set_servo_pulsewidth(6,0)
os.system('pigs s 16 1500')

#Inicializar Modelo
def cargar():
	json_file = open('/home/pi/Cmodels.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("/home/pi/Cmodels.h5")
	return loaded_model

loaded_model = cargar()
loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy']) 

#Funcion Prediccion
def get_direction(image):
	res = loaded_model.predict(image.reshape((-1,240,320,3)), batch_size=1)
        #print(res)
        move = np.argmax(res)
        direction = []
        if move == 0:
            direction = 1
        elif move == 1:
            direction = 0
        elif move == 2:
            direction = 2
        return direction

#Esperar antes de empezar procesamiento
time.sleep(0.1)

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    frame1=frame.array
    frame2= frame1.copy()
    width = frame1.shape[1]
    height = frame1.shape[0]

    #Enviar Imagen al Servidor
    result, frame = cv2.imencode('.jpg', frame1, encode_param)
    data = pickle.dumps(frame, 0)
    size = len(data)
    client_socket.sendall(struct.pack(">L", size) + data)
    img_counter += 1

    #Reconocer Imagen Keras---Direccion
    res = loaded_model.predict(frame2.reshape((-1,240,320,3)), batch_size=1)

    move = np.argmax(res)
    estado=""

    #Recibir datos servidor
    datos = client_socket.recv(4096)
    datos_e = datos.decode('utf8') 
    spl_d = datos_e.split(',')
    ymin = float(spl_d[0])
    xmin = float(spl_d[1])
    ymax = float(spl_d[2])
    xmax = float(spl_d[3])
    detectado = str(spl_d[4])
    sign = str(spl_d[5])
    dist = float(spl_d[6])
    print(sign)
    cv2.rectangle(frame2,(int(xmin),int(ymax)),(int(xmax),int(ymin)),(0,255,0),3)
    if(sign == "Stop" and dist < 12 ):
	print('Stop a:',dist)
	print('Parar Stop 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    elif(sign == "Yield" and dist < 12 ):
	print('Yield a:',dist)
	print('Esperar Yeild 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3) 
    elif(sign == "Priority road" and dist < 12 ):
	print('Priority road a:',dist)
	print('Parar Priority road 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    elif(sign == "Speed limit (50km/h)" and dist < 12 ):
	print('Speed limit (50km/h) a:',dist)
	print('Speed limit (50km/h) 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    elif(sign == "Ahead only" and dist < 12 ):
	print('Ahead only a:',dist)
	print('Parar Ahead only 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    elif(sign == "End of no passing" and dist < 12 ):
	print('End of no passing a:',dist)
	print('Parar End of no passing 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    elif(sign == "No passing" and dist < 12 ):
	print('No passing a:',dist)
	print('Parar No passing 3s')
	os.system('pigs s 16 0')
	if len(sign) != 0:
            cv2.putText(frame2, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
	time.sleep(3)
    else:
	#Sino simplemente controlamos direccion segun resultado keras
    	print('Ninguna Senal')
	if (move == 2 ):
       		if(estado!="derecha"):
               		pi.set_servo_pulsewidth(6, 1560)
               		time.sleep(0.3)
	       		pi.set_servo_pulsewidth(6,0)
	       		os.system('pigs s 16 1595')
	       		time.sleep(0.18)
	       		os.system('pigs s 16 1500')
	       		time.sleep(0.1)
               		estado="derecha"
			print('derecha')
	       		cv2.putText(frame2,"Derecha->", (10,150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA)

   	elif (move == 1):
   	    	if(estado!="izquierda"):
	       		pi.set_servo_pulsewidth(6, 1930)
	       		time.sleep(0.3)
	       		pi.set_servo_pulsewidth(6,0)
               		os.system('pigs s 16 1595')
	       		time.sleep(0.18)
	       		os.system('pigs s 16 1500')
	       		time.sleep(0.1)
       	       		estado="izquierda"
			print('izquierda')
	       		cv2.putText(frame2,"<-Izquierda", (220, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA)

   	else:
      		if(estado!="recto"):
	       		pi.set_servo_pulsewidth(6, 1790)
	       		time.sleep(0.3)
	       		pi.set_servo_pulsewidth(6,0)
	       		os.system('pigs s 16 1595')
	      	 	time.sleep(0.18)
	       		os.system('pigs s 16 1500')
	       		time.sleep(0.1)
       	       		estado="recto"
			print('recto')
	       		cv2.putText(frame2,"Recto", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA)
      

    #Mostrar resultados
    cv2.imshow('Conduccion Autonoma',frame2)
    #cv2.imshow('ImageWindow',crop)
    del datos
    del frame2
    cv2.waitKey(1)
    rawCapture.truncate(0)