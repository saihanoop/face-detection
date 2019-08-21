import cv2
import tensorflow as tf
import numpy as np


path = "C:/Users/Hanoop/AppData/Local/Programs/Python/Python36/ps/dataset/test_0.jpg"
model = tf.keras.models.load_model("faces_2.model")

img_array = cv2.imread(path,0)
#img_array = (img_array*1.25)/100        
new_array = cv2.resize(img_array,(200,200))
cv2.imshow('img',new_array)
new_array = new_array.reshape(-1,200,200,1)


prediction = model.predict(new_array)

file_name = path.split('/')[-1]

print(file_name,"->",prediction)
print("-----------",np.where(prediction==np.amax(prediction))[1][0],"-----------")
            
