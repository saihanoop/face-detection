import cv2
import os

disk = "D:/109CANON"
folders = ["gou","han","mani"]
count = 0
for name in folders:
  path = disk + "/"+name  
  for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path,img), 0)
        cv2.imwrite('dataset/face-{c}_{n}.jpg'.format(c=count,n=folders.index(name)),img_array)
        count+=1
        new = cv2.flip(img_array, 1)
        cv2.imwrite('dataset/face-{c}_{n}.jpg'.format(c=count,n=folders.index(name)),new)
        count+=1
        cv2.imwrite('dataset/face-{c}_{n}.jpg'.format(c=count,n=folders.index(name)),img_array)
        count+=1
        new = cv2.flip(img_array, 1)
        cv2.imwrite('dataset/face-{c}_{n}.jpg'.format(c=count,n=folders.index(name)),new)
        count+=1

  print("folder done")  

print("done")
        
