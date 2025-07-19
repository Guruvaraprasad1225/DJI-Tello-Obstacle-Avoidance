import cv2
VideoAccess = cv2.VideoCapture(0)
ret, frame = VideoAccess.read()
cv2.imshow("Webcam",frame)
if cv2.waitkey(1)==ord('q'):
    break
VideoAccess.release()
cv2.destroyAllWindows()