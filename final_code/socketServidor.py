#!/usr/bin/python
#Importar Librerias
import socket
import sys
import cv2
import csv
import pickle
import numpy as np
import struct ## new
import zlib
import os
import tensorflow as tf
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from skimage import color, exposure, transform
from keras.models import model_from_json
from keras.optimizers import SGD
from keras.utils.np_utils import to_categorical
sys.path.append("..")

# Import utilidades framework API Tensorflow ObjectDetection
from utils import label_map_util
from utils import visualization_utils as vis_util

#Cargar modelo keras
def cargar():
	json_file = open('C:/tensorflow1/models/research/object_detection/ke/kmodels.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	loaded_model.load_weights("C:/tensorflow1/models/research/object_detection/ke/kmodels.h5")
	return loaded_model

#Cargar etiquetas senal
def etiqueta():
	with open('C:/tensorflow1/models/research/object_detection/ke/signnames.csv','r')as f:
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

#Funcion calcular distancia
def distancia (ancho, focal, perAncho):
	return (ancho*focal)/perAncho

#Funcion encontrar senal
def detectar(img):
	gris = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	gris = cv2.GaussianBlur(gris,(5,5),0)
	canny = cv2.Canny(gris,35,125)
	contorno,hi =  cv2.findContours(canny.copy(),cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	c = max(contorno,key = cv2.contourArea)
	return cv2.minAreaRect(c)
    
#Variables 
know_D = 5.11811
know_W = 1.81165
#focal_length = 2228.57578287
focal_length = 2592*3.6/3.67

#Inicializar variables y cargar modelo
lista = etiqueta()
loaded_model = cargar()
sd = SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
loaded_model.compile(loss="categorical_crossentropy", optimizer=sd, metrics=["accuracy"]) 

#Inicializar variables API 
MODEL_NAME = 'inference_graph'
CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','label_map.pbtxt')
NUM_CLASSES = 1
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

#Cargar Grafo y tensores(identificadores)
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

#Conexion socket con RaspBerry
serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
serv.bind(('192.168.2.149', 8880))
serv.listen(5)

while True:
    conn, addr = serv.accept()
    from_client = ''
    data = b""
    payload_size = struct.calcsize(">L")
    
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
       
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame=pickle.loads(frame_data, fix_imports=True, encoding="bytes")
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        frame_expanded = np.expand_dims(frame, axis=0)

        #Detectar Objectos--Senal
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        #Inicializar variables datos-enviar
        width = frame.shape[1]
        height = frame.shape[0]
        xmin = 0
        ymin = 0
        xmax = 0
        ymax = 0
        detect = False
        sign = ""
        dist = 0
        for i, box in enumerate(np.squeeze(boxes)):
            if(np.squeeze(scores)[i] > 0.8):
                xmin = box[1]*width
                ymin = box[0]*height
                xmax = box[3]*width
                ymax = box[2]*height
                detect = True 
        
        if(width!=0.0 and height!=0.0 and ymin!=0.0 and xmin!=0.0 and xmax!=0.0 and ymax!=0.0):
            #Si hay objeto-senal lo reconocemos con keras
            crop1 = frame[int(ymin)-3:int(ymax)+3,int(xmin)-3:int(xmax)+3]
            crop = np.copy(crop1) 
            width1 = crop.shape[1]
            height1 = crop.shape[0]
            if(width1!=0.0 and height1!=0.0):
                max_index = np.argmax(loaded_model.predict(prep(crop)))
                sign = lista[max_index]
                marker = detectar(crop1)
                #*2.54
                #Aproximacion distancia
                dist = (distancia(know_W,focal_length,crop.shape[1])/12)*2.54
            #print(dist)
            #print(max_index)
            #print(lista[max_index])
        else:
            crop = np.zeros((100,100,3),np.uint8)
            
        #Encapsulamos y enviamos RaspBerry
        da = str(ymin)+","+str(xmin)+","+str(ymax)+","+str(xmax)+","+str(detect)+","+str(sign)+","+str(dist)
        conn.send(da.encode('utf8'))
       
       #Mostramos resultados
        cv2.rectangle(frame,(int(xmin),int(ymax)),(int(xmax),int(ymin)),(0,255,0),3)
        if len(sign) != 0:
            cv2.putText(frame, str(sign), (int(xmin-30), int(ymax+20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), lineType=cv2.LINE_AA) 
        cv2.imshow('Senal',frame)
        cv2.waitKey(1)