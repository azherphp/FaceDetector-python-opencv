import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

image=cv2.imread("sumeet.jpg")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_image,
scaleFactor=1.1,minNeighbors=5)

for x,y,w,h in faces:
    image=cv2.rectangle(image,
    (x,y),(x+w,y+h),(0,255,0),3)

resized_img=cv2.resize(image,(int(image.shape[1]/5),int(image.shape[0]/5)))

cv2.imshow("Gray", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
