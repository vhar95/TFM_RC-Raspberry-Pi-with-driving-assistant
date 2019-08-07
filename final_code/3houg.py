#!/usr/bin/python
#Importar Librerias (Camara,Opencv,Math,Pigpio)
from picamera.array import PiRGBArray
import pigpio
from picamera import PiCamera
import time
import cv2
import numpy as np
import math

#Inicializar variables y Pin 6 servo
theta=0
minLineLength = 5
maxLineGap = 10
pi = pigpio.pi()
pi.set_mode(6,pigpio.OUTPUT)
pi.set_servo_pulsewidth(6,0)

#Inicializar Camara
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = PiRGBArray(camera)

#Esperar antes de empezar procesamiento
time.sleep(0.1)

#Captura Frames Stream Camara
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
   image = frame.array
   
   #Cambiar espacio color a Gris
   gris = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   
   #Aplicar Filtro Guassiano
   blur = cv2.GaussianBlur(gris, (5, 5), 0)
   
   #Detectamos esquinas/lineas (Canny y tranformada de hough)
   canny = cv2.Canny(blur, 85, 85)
   lineas = cv2.HoughLinesP( canny,1,np.pi/180,10,minLineLength,maxLineGap)

   #Analizamos valor de la transformada
   if(lineas.all() != None):
       for x in range(0, len(lineas)):
           for x1,y1,x2,y2 in lineas[x]:
               cv2.line(image,(x1,y1),(x2,y2),(0,255,0),2)
               theta=theta+math.atan2((y2-y1),(x2-x1))

   img2 = cv2.resize(image.copy(), dsize=(450, 400), interpolation=cv2.INTER_CUBIC)
   
   #Establecer umbral de derecha y izquierda segun pruebas sobre el circuito
   umbral=6

   #Segun el resultado mandara un pulso o otro al pin 6
   estado=""
   if(theta>umbral):
	if(estado!="izquierda"):
	       pi.set_servo_pulsewidth(6, 1930)
	       time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
       	   estado="izquierda"
	       cv2.putText(img2,"<-Izquierda", (260, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)

   if(theta<-umbral):
	if(estado!="derecha"):
	       pi.set_servo_pulsewidth(6, 1560)
           time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
           estado="derecha"
	       cv2.putText(img2,"Derecha->", (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)

   if(abs(theta)<umbral):
	if(estado!="recto"):
	       pi.set_servo_pulsewidth(6, 1790)
	       time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
       	   estado="recto"
	       cv2.putText(img2,"Recto", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)
   theta=0
   
   #Ventana con los resultados
   cv2.imshow("CannyLines",img2)
   del(img2)
  
   key = cv2.waitKey(1) & 0xFF
   rawCapture.truncate(0)
   if key == ord("q"):
       break