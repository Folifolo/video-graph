import cv2
import os

print(cv2.__version__)
name = 'sample.avi'

if (os.path.isfile (name)):
   size= os.path.getsize(name)

cap = cv2.VideoCapture(name)

while(cap.isOpened()):
   ret, frame = cap.read()
   if ret != True:
       break
   gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   cv2.imshow('frame',gray)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break

cap.release()
cv2.destroyAllWindows()