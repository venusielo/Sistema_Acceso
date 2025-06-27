from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import facenet
import detect_face
import os
import time
import pickle
from PIL import Image
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import imutils
from relay import *
from FormatoLetras import *
from señales import *
import sys
import types

# Crear un módulo falso para redirigir la importación antigua a la nueva ruta
import sklearn.svm
sys.modules['sklearn.svm.classes'] = types.ModuleType('classes')
sys.modules['sklearn.svm.classes'].SVC = sklearn.svm.SVC
sys.modules['sklearn.svm.classes'].NuSVC = sklearn.svm.NuSVC
sys.modules['sklearn.svm.classes'].SVR = sklearn.svm.SVR
sys.modules['sklearn.svm.classes'].NuSVR = sklearn.svm.NuSVR
sys.modules['sklearn.svm.classes'].OneClassSVM = sklearn.svm.OneClassSVM
sys.modules['sklearn.svm.classes'].LinearSVC = sklearn.svm.LinearSVC
sys.modules['sklearn.svm.classes'].LinearSVR = sklearn.svm.LinearSVR


det1 = np.load('./npy/det1.npy', allow_pickle=True, encoding='latin1')
det2 = np.load('./npy/det2.npy', allow_pickle=True, encoding='latin1')
det3 = np.load('./npy/det3.npy', allow_pickle=True, encoding='latin1')

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(
        gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess, npy)
        minsize = 30  # minimum size of face
        threshold = [0.7, 0.72, 0.75]  # three steps' threshold
        factor = 0.709  # scale factor
        margin = 44

        batch_size = 100  # 1000
        image_size = 182
        input_image_size = 160
        HumanNames = os.listdir(train_img)
        HumanNames.sort()
        facenet.load_model(modeldir)
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        classifier_filename_exp = os.path.expanduser(classifier_filename)

        with open(classifier_filename_exp, 'rb') as infile:
            (model, class_names) = pickle.load(infile, encoding='latin1')

            # Variables para que una persona no abra dos veces la puerta en una sola vez
            i = 0
            j = 0
            dsize = (1100, 1100)  # tamaño de la interfaz de la cámara
            while True:
                cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
                # detector = HandDetector(detectionCon=0.8, maxHands=2)
                i = 0
                while cap.isOpened():
                    succes, img = cap.read()
                    img = cv2.resize(img, dsize)
                    img = cv2.flip(img, 1)

                    # hands, img = detector.findHands(img, flipType=True)
                    # Esto es para que el nombre de la persona aparezca 10 veces que se lea un frame
                    if j > 10:
                        j = 0
                    elif j != 0:
                        j += 1
                        cv2.putText(img, 'Hi ' + '{}'.format(nombre) + "!", org, font, fontScale, color, thickness, cv2.LINE_AA)
                    ###

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
                            try:
                                # Validar que la cara esté dentro de los límites de la imagen
                                if xmin <= 0 or ymin <= 0 or xmax >= len(img[0]) or ymax >= len(img):
                                    # print('Face is very close!')
                                    continue  # Saltar esta cara si está fuera de rango

                                # Recortar y preparar la imagen para el embedding
                                cropped_face = img[ymin:ymax, xmin:xmax, :]
                                cropped_face = facenet.flip(cropped_face, False)
                                scaled_face = np.array(Image.fromarray(cropped_face).resize((image_size, image_size)))
                                scaled_face = cv2.resize(scaled_face, (input_image_size, input_image_size), interpolation=cv2.INTER_CUBIC)
                                scaled_face = facenet.prewhiten(scaled_face)
                                scaled_reshape = scaled_face.reshape(-1, input_image_size, input_image_size, 3)

                                feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                                emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)

                                predictions = model.predict_proba(emb_array)
                                best_class_indices = np.argmax(predictions, axis=1)
                                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

                                if best_class_probabilities > 0.19:  # Threshold configurable
                                    result_name = HumanNames[best_class_indices[0]]
                                    #persona = (result_name, fingers1 + fingers2)  # Asegúrate que fingers1 y fingers2 estén definidos
                                    nombre = result_name

                                    cv2.putText(img, 'Hola, ' + nombre + "!", (xmin, ymin - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    
                                    #decir_nombre(nombre)

                                    if persona in señales.items():
                                        j += 1
                                        nombre = result_name
                                        if abrirPuerta == False and nombre != anterior:
                                            anterior = result_name
                                            on()  # Función para abrir la puerta
                                            abrirPuerta = True
                                else:
                                    pass
                            except Exception as e:
                                # Opcional: imprimir el error para depuración
                                # print(f"Error procesando cara: {e}")
                                pass

                    cv2.imshow('', img)

                    key = cv2.waitKey(1)
                    if key == 113:  # Código ASCII para 'q'
                        break

                cap.release()
                cv2.destroyAllWindows()
                break  # Salir del while True después de cerrar la cámara
