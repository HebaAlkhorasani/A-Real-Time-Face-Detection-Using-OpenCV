import cv2

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') #To load the classfier

image =cv2.imread("faces2.jpg") #open the image

image2gray= cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #image to gray scale

detection = faceCascade.detectMultiScale(image2gray, 1.09, 5) #used to detect faces

for (x, y, w, h) in detection: #draw the rectangles around the detected faces
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
    
cv2.imshow('detected_faces', image)
cv2.waitKey()





