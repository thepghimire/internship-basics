import datetime #for datetime in video feed
import imutils #opencv package for mathematical operations
import cv2

def pre_process(frame):
    frame = imutils.resize(frame, width=500)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21,21), 0)
    return gray

def find_motion(current_frame, first_frame):
    frame_delta = cv2.absdiff(first_frame, current_frame)
    # absdiff for detecting motion. First frame has environment. 
    # absdiff le background ra foreground separate garcha. so, can only detect
    # foreground movements.
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    return thresh

def draw_in_image(image):
    cv2.imshow("Frame", image)

def draw_motion(thresh, frame):
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #contorus nikalyo
    cnts = imutils.grab_contours(cnts)
    for c in cnts:
        if cv2.contourArea(c) < 100:
            continue
        (x,y,w,h) = cv2.boundingRect(c) #THIS MAY BE GALAT
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 3)
        cv2.putText(frame, datetime.datetime.now().strftime("%d %M %Y %I:%M:%S"),
                    (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35,(0,0,255), 1)
        cv2.namedWindow("Motion Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Motion_Detection", frame)
        cv2.namedWindow("Thresh", cv2.WINDOW_NORMAL)
        cv2.imshow("Threshold_detection", thresh)

def detect_motion(video_file):
    vs = cv2.VideoCapture(video_file)
    first_frame = None
    while True:
        status, frame = vs.read()
        if not status:
            break

        gray_frame = pre_process(frame)
        frame = imutils.resize(frame, width=500)
        if first_frame is None:
            first_frame = gray_frame
            continue

        motion_image = find_motion(gray_frame, first_frame)
        draw_motion(motion_image, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
    
if __name__ == "__main__":
    video_file = 0 #webcam
    detect_motion(video_file)








    