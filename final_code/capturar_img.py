#!/usr/bin/python
#Importar Librerias
from picamera.array import PiRGBArray
from picamera import PiCamera
import cv2
import pigpio
import os
import time

FRAME_H =  240
FRAME_W = 320

FRAME_RATE = 30
TRIAL_NUM  = 0

for file in os.listdir('.'):
    if 'drive_trial_' in file:
        TRIAL_NUM += 1

#Configurar Camara
camera = PiCamera()
camera.resolution = (FRAME_W, FRAME_H)
camera.framerate = FRAME_RATE
rawCapture = PiRGBArray(camera, size=(FRAME_W, FRAME_H))
time.sleep(0.1)

#Inicializar Pin 6 Servo
pi = pigpio.pi()
pi.set_mode(6,pigpio.OUTPUT)
pi.set_servo_pulsewidth(6,0)

trial = 'drive_trial_' + str(TRIAL_NUM)
os.system('mkdir ' + trial)
os.system('mkdir ' + trial + '/image')
os.system('mkdir ' + trial + '/image/right')
os.system('mkdir ' + trial + '/image/left')
os.system('mkdir ' + trial + '/image/str')
counter = 0

#Captura Frames Stream Camara
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    
    image = frame.array 
	#Segun el comando mandara un pulso o otro al pin 6 y guardar imagen
    print('Introduzca comando')
    tecla = int(input())
    
    if (tecla == 0):
        print('Exiting...')
        break

    elif (tecla == 8):
        print('up')  
        pi.set_servo_pulsewidth(6, 1790)
		time.sleep(0.3)
		pi.set_servo_pulsewidth(6,0)
		time.sleep(0.05)
        os.system('pigs s 16 1596')
		time.sleep(0.2)
		os.system('pigs s 16 1500')
		time.sleep(0.05)   
        cv2.imwrite(trial + '/image/str/w' + str(counter).zfill(6) + '.png', image)
    elif (tecla == 2):
        print('down')       
        cv2.imwrite(trial + '/image/' + str(counter).zfill(6) + '.png', image)
    elif (tecla == 4):
        print('left')
		pi.set_servo_pulsewidth(6, 1930)
		time.sleep(0.3)
		pi.set_servo_pulsewidth(6,0)
		time.sleep(0.05)
        os.system('pigs s 16 1596')
		time.sleep(0.2)
		os.system('pigs s 16 1500')
		time.sleep(0.05) 
        cv2.imwrite(trial + '/image/left/l' + str(counter).zfill(6) + '.png', image)
    elif (tecla == 6):
        print('right') 
        pi.set_servo_pulsewidth(6, 1560)
        time.sleep(0.3)
		pi.set_servo_pulsewidth(6,0)
        time.sleep(0.05)
        os.system('pigs s 16 1596')
		time.sleep(0.2)
		os.system('pigs s 16 1500')
		time.sleep(0.05) 
        cv2.imwrite(trial + '/image/right/r' + str(counter).zfill(6) + '.png', image)
    else:
        print('Error Comando')          
          
    counter += 1

    cv2.imshow("Frame", image)
    rawCapture.truncate(0)
    time.sleep(0.2)
    key = cv2.waitKey(1) & 0xFF
    