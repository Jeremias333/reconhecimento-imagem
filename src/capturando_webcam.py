import cv2
import time
#cv2.VideoCapture("videoFilePath")

webcamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)#setando a camera integrada do pc
classificador_face = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

time.sleep(2)

while True:
	camera, frame = webcamera.read()

	face_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	detecta = classificador_face.detectMultiScale(face_cinza)

	for(x, y, l, a) in detecta:
		cv2.rectangle(frame, (x, y), (x+l, y+a), (255, 0, 0), 2)

	cv2.imshow("imagem webcamera", frame)

	if cv2.waitKey(1) == ord('q'):
		break

webcamera.release()
cv2.destroyAllWindows()