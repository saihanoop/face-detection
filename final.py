import numpy as np
import cv2
import tensorflow as tf
import numpy as np
# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

face_cascade = cv2.CascadeClassifier('cascade_face.xml')
model = tf.keras.models.load_model("faces_2.model")
name = ['goutham','hanoop','maniketh']
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    img = cv2.flip(img,+1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face = face_cascade.detectMultiScale(gray)
    for (x,y,w,h) in face:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,200,0),2)
        img_array = gray[y:y+h+10, x:x+w]
        show_array = cv2.resize(img_array,(200,200))
        new_array = show_array.reshape(-1,200,200,1)
        prediction = model.predict(new_array)
        print(prediction)
        #cv2.imshow(str(prediction),show_array)
        cv2.putText(img,str(name[np.where(prediction==np.amax(prediction))[1][0]]), (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
        
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
