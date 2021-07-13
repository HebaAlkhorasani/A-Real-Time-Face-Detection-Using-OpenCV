import cv2
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

while True:
    _, video = cap.read()
    video2gray= cv2.cvtColor(video, cv2.COLOR_BGR2GRAY) #image to gray scale

    detection = faceCascade.detectMultiScale(video2gray, 1.09, 5) #used to detect faces

    for (x,y,w,h) in detection: #draw the rectangles around the detected faces
        cv2.rectangle(video,(x,y),(x+w,y+h),(0,255,0),3)
        
    cv2.imshow('detected_faces',video)
    k = cv2.waitKey(30) & 0xff
    if k == 27: # click 'ESC' to stop
        break
        
cap.release()
cv2.destroyAllWindows()