import cv2
import os

# простейшая работа с видео
def print_video_info(videoName):
    if os.path.isfile(name):
        size = os.path.getsize(name)
        print 'video size is ' + str(size) + ' bytes'
    else:
        print 'file doesn\'t exist'
        return

def show_video_grayscale(videoName):
    if not os.path.isfile(name):
        print 'file doesn\'t exist'
        return

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


###
name = 'sample.avi'
print 'opencv version: ' + cv2.__version__

print_video_info (name)
show_video_grayscale(name)