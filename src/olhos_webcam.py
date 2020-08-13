import cv2
import time
#cv2.VideoCapture("videoFilePath")
webcamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)#setando a camera integrada do pc
time.sleep(2)

while True:
	camera, frame = webcamera.read()
	cv2.imshow("imagem webcamera", frame)

	if cv2.waitKey(1) == ord('q'):
		break

webcamera.release()
cv2.destroyAllWindows()