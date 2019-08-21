import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

path = "C:/Users/Hanoop/AppData/Local/Programs/Python/Python36/ps/resized"
training_data =[]
person = ['goutham','hanoop','maniketh']
for img in os.listdir(path):  
              img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)  
              training_data.append([img_array, img.split('_')[1][0]])
              
print("length:",len(training_data))
random.shuffle(training_data)    

for k in training_data[:10]:
    print(k[1])


x=[]
y=[]

for features,name in training_data[:8000]:
    x.append(features)
    y.append(name)
    

x = np.array(x).reshape(-1, 200, 200, 1)
x=x/255.0
print(x[1])
import pickle

pickle_out = open("x.pickle","wb") 
pickle.dump(x, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb") 
pickle.dump(y, pickle_out)
pickle_out.close()

