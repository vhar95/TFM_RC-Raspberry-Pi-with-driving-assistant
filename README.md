# TFM - RC Raspberry Pi with driving assistant

The main objective that has been proposed in this TFM is the creation of a radio control (RC) car in 3D printing, which is capable of autonomous driving in a small circuit. Also this car will have the support of a driving assistant where this assistant can detect and classify the signals and their distance to the car. All this in order to understand the basis of autonomous driving and main problems and advantages.

RC Car Features:
- Manual user control
- Real-time video streaming
- Identify circuit lines, to act later on the direction of the car and thus be able to follow a circuit autonomously
- Identification of traffic signs, thus performing the task of driving assistant

![Esquema](/img/8.png)

![Esquema](/img/10.png)

## Node-red flow - Manual User Control

![Esquema](/img/1.jpg)

## Keras Model Servo Direction

At this point a model will be built in keras so that it is able to detect the direction that the RC car should take according to the classification of the images.
You can follow the development in the notebook of this repository:[notebook](/servo_position_classification/ModeloKerasDireccion.ipynb)

### Keras

![Esquema](/img/4.png)

### Hough transform

![Esquema](/img/2.png)

### Histogram analysis

![Esquema](/img/3.png)

## Signal Detection and Classification


At this point the detection and classification of traffic signals will be developed, using 3 methods.


### Haar Cascades

You can follow the development in the notebook of this repository:[notebook](/signal_detection_and_classification/haar_cascades/HaarCascadeStop.ipynb)

![Esquema](/img/5.png)

### Tensorflow Object Detection API

You can follow the development in the notebook of this repository:[notebook](/signal_detection_and_classification/tensorflow_api/TensorflowObjetoSenal.ipynb)

![Esquema](/img/6.png)

### Keras Signal Classification

You can follow the development in the notebook of this repository:[notebook](/signal_detection_and_classification/keras_signal_classification/ReconocerSenales.ipynb)

![Esquema](/img/7.png)

## Final Result

![Esquema](/img/9.png)


<figure class="video_container">
  <iframe src="https://drive.google.com/file/d/1YEKs96VWj8FArCZnf8O0xklYsutgF6pk/view" frameborder="0" allowfullscreen="true"> </iframe>
</figure>