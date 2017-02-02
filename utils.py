import cv2
import os

def print_video_info(videoName):
    if os.path.isfile(videoName):
        size = os.path.getsize(videoName)
        print 'video size is ' + str(size) + ' bytes'
    else:
        print 'file doesn\'t exist'
        return



def show_video_grayscale(videoName):
    if not os.path.isfile(videoName):
        print 'file doesn\'t exist'
        return

    cap = cv2.VideoCapture(videoName)

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