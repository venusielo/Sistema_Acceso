import cv2


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