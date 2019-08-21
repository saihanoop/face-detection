"""import cv2
import os

path = "C:/Users/Hanoop/AppData/Local/Programs/Python/Python36/ps/dataset"
for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), 0)
        new_array = cv2.resize(img_array,(200,200))
        cv2.imwrite("resized/"+img,new_array)  

print("done")
        
