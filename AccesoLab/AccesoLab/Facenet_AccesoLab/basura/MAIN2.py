from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cv2
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
import cv2
import os
import time
from cvzone.HandTrackingModule import HandDetector
import imutils
import numpy as np
from relay import *

#SEÑALES DE CADA UNO
señales= {'Diana':[1,1,1,1,1,1,1,1,1,1], 
            'Martin':[0,0,0,0,0,0,0,0,0,0],
                'Eliud':[1,1,0,0,0,1,1,0,0,0],
                    'Abraham':[1,0,0,0,0,1,0,0,0,0],
                        'Vesna':[0,1,1,0,0,0,1,1,0,0],
                            'Juventino':[1,1,1,0,0,1,1,1,0,0],
                                'Rodolfo':[1,0,0,0,1,1,0,0,0,1],
                                    'Jorge':[1,1,0,0,1,1,1,0,0,1],
                                        'Jesus':[0,1,1,0,0,0,1,1,0,0]
}

cropped = []
scaled = []
scaled_reshape = []

# Modificaciones para texto que aparece en la pantalla
# font
font = cv2.FONT_HERSHEY_SIMPLEX 
# org
org = (470, 100)
# fontScale
fontScale = 2
# Blue color in BGR
color = (0, 0, 0)
# Line thickness of 2 px
thickness = 3
######################################################
modeldir = './model/20180402-114759.pb'
classifier_filename = './class/classifier.pkl'
npy = './npy'
train_img = "./train_img"

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7, 0.8, 0.8]  # three steps's threshold
        factor = 0.709  # scale factor
        margin = 44
        batch_size = 100  # 1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        # print('Loading Model')
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)
        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')
            # Variables que se ocupan para que una persona no abra dos veces la puerta en una sola vez
            i=0
            j=0
            dsize= (1100,1100) #   tamaño  de la interfaz de la camara
            while True:
                cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
                detector = HandDetector(detectionCon=0.8, maxHands=2)
                i=0
                while (cap.isOpened()):
                    succes, img = cap.read()
                    img = cv2.resize(img, dsize)
                    img = cv2.flip(img,1)
                    hands, img = detector.findHands(img)
                    if j>10:
                        j=0
                    elif j != 0:
                        j=j+1
                        cv2.putText(img, 'Hi '+'{}'.format(nombre)+ "!", org, font, fontScale, color, thickness, cv2.LINE_AA)
                    if not hands:
                        anterior = None
                        abrirPuerta = False
                    if hands:
                        hand1 = hands[0]
                        fingers1 = detector.fingersUp(hand1) # detecta el numero de dedos como una cadena de 0 y 1 (0 si esta agachado y 1 levantado)
                        if len(hands) == 2:
                            hand2 = hands[1]
                            fingers2 = detector.fingersUp(hand2)
                            if fingers1 + fingers2 in señales.values():
                                if img.ndim == 2:
                                    img = facenet.to_rgb(img)
                                bounding_boxes, _ = detect_face.detect_face(
                                    img, minsize, pnet, rnet, onet, threshold, factor)
                                faceNum = bounding_boxes.shape[0]
                                if faceNum > 0:
                                    det = bounding_boxes[:, 0:4]
                                    img_size = np.asarray(img.shape)[0:2]
                                    for i in range(faceNum):
                                        emb_array = np.zeros((1, embedding_size))
                                        xmin = int(det[i][0])
                                        ymin = int(det[i][1])
                                        xmax = int(det[i][2])
                                        ymax = int(det[i][3])
                                        # try:
                                        #     # inner exception
                                        #     if xmin <= 0 or ymin <= 0 or xmax >= len(img[0]) or ymax >= len(img):
                                        #         # print('Face is very close!')
                                        #         continue
                                        cropped.append(img[ymin:ymax, xmin:xmax, :])
                                        cropped[i] = facenet.flip(cropped[i], False)
                                        scaled.append(np.array(Image.fromarray(
                                            cropped[i]).resize((image_size, image_size))))
                                        scaled[i] = cv2.resize(scaled[i], (input_image_size, input_image_size),
                                                            interpolation=cv2.INTER_CUBIC)
                                        scaled[i] = facenet.prewhiten(scaled[i])
                                        scaled_reshape.append(
                                            scaled[i].reshape(-1, input_image_size, input_image_size, 3))
                                        feed_dict = {
                                            images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                                        emb_array[0, :] = sess.run(
                                            embeddings, feed_dict=feed_dict)
                                        predictions = model.predict_proba(emb_array)
                                        best_class_indices = np.argmax(predictions, axis=1)
                                        best_class_probabilities = predictions[np.arange(
                                            len(best_class_indices)), best_class_indices]
                                        cv2.rectangle(img, (xmin, ymin),(xmax, ymax), (0, 255, 0), 2)
                                        # print(best_class_probabilities)
                                        if best_class_probabilities > 0.19: 
                                            result_name=HumanNames[best_class_indices[0]]
                                            persona = (result_name, fingers1+fingers2)
                                            if persona in señales.items():
                                                    j=j+1
                                                    nombre= result_name
                                                    if abrirPuerta == False and nombre != anterior:
                                                        anterior = result_name
                                                        print("hola")
                                                        # on()
                                                        abrirPuerta=True
                                        else:
                                            pass
                                    #except:
                                    #     pass
                    cv2.imshow('', img)


                    key = cv2.waitKey(1)
                    if key == 113:  # "q"
                        break
        cap.release()
        cv2.destroyAllWindows()