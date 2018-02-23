import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

camera_port = 1
ramp_frames = 30

camera = cv2.VideoCapture(camera_port)

def get_image():
    retval, im = camera.read()
    return im


for i in range(ramp_frames):
    temp = get_image()
print("Taking image...")
camera_capture = get_image()
file = "test_image.png"
cv2.imwrite(file, camera_capture)

image=cv2.imread("test_image.png")
gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

faces=face_cascade.detectMultiScale(gray_image,
scaleFactor=1.1,minNeighbors=5)

for x,y,w,h in faces:
    image=cv2.rectangle(image,
    (x,y),(x+w,y+h),(0,255,0),3)

resized_img=cv2.resize(image,(int(image.shape[1]/2),int(image.shape[0]/2)))

cv2.imshow("Gray", resized_img)
while True:
    key = cv2.waitKey(1)
    if key == ord('q'):
        exit()
cv2.waitKey(0)
cv2.destroyAllWindows()


