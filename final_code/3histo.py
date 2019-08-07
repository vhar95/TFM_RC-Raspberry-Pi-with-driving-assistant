#!/usr/bin/python
#Importar Librerias (Camara,Opencv,Math,Pigpio)
import picamera.array
from picamera import PiCamera
import time
import cv2
import numpy as np
import math
import pigpio
import plotly.plotly as py
import plotly.tools as tls
from pylab import *
from moviepy.video.io.bindings import mplfig_to_npimage

#Inicializar Camara
camera = PiCamera()
camera.resolution = (320, 240)
camera.framerate = 30
rawCapture = picamera.array.PiRGBArray(camera)

#Inicializar Pin 6 Servo
pi = pigpio.pi()
pi.set_mode(6,pigpio.OUTPUT)
pi.set_servo_pulsewidth(6,0)

#Inicializar Grafica Histograma
fig, ax = subplots(figsize=(4,3),facecolor='none', edgecolor='none')
line, = ax.plot(np.arange(320))
ax.set_facecolor((0,0,0,0))
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)
ax.set_frame_on(False)
line.set_color("green")
fig.set_alpha(0)
xlim([0,360]) 
ylim([0,400])
box('off')
tight_layout()
graph = mplfig_to_npimage(fig)
gh,g2,_ = graph.shape

#Esperar antes de empezar procesamiento
time.sleep(0.1)

#Captura Frames Stream Camara
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
   
   #Cambiar espacio color a YUV
   imgl = cv2.cvtColor(frame.array,cv2.COLOR_BGR2YUV)
   img = imgl[:,:,0]

   #Transformar imagen binario
   min = 0        
   max = 90  
   binario = np.zeros_like(img)        
   binario[(img >= min) & (img<= max)] = 1
   histograma = np.sum(binario[binario.shape[0]//2:,:], axis=0)

   #Representamos histograma
   line.set_ydata(histograma)

   #Segun que parte del histograma tenga mayor valor, indicara derecha, izquierda o recto
   derecha = np.sum(histograma[0:150], dtype=int)
   izquierda = np.sum(histograma[-150:], dtype=int)
   img2 = cv2.resize(frame.array.copy(), dsize=(450, 400), interpolation=cv2.INTER_CUBIC)
   
   #Segun el resultado mandara un pulso o otro al pin 6
   estado=""
   if (derecha > izquierda):
       if(estado!="derecha"):
               pi.set_servo_pulsewidth(6, 1560)
               time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
               estado="derecha"
	       cv2.putText(img2,"Derecha->", (0, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA) 

   elif (derecha < izquierda):
       if(estado!="izquierda"):
	       pi.set_servo_pulsewidth(6, 1930)
	       time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
       	       estado="izquierda"
	       cv2.putText(img2,"<-Izquierda", (260, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)

   else:
      if(estado!="recto"):
	       pi.set_servo_pulsewidth(6, 1790)
	       time.sleep(0.3)
	       pi.set_servo_pulsewidth(6,0)
       	       estado="recto"
	       cv2.putText(img2,"Recto", (150, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), lineType=cv2.LINE_AA)
      
   #Ventana con los resultados
   cv2.imshow("Camera-Histograma",img2)
   cv2.imshow("Histograma",mplfig_to_npimage(fig))
   del(img2)

   key = cv2.waitKey(1) & 0xFF
   rawCapture.truncate(0)
   if key == ord("q"):
       break
       
