import cv2
import time
#cv2.VideoCapture("videoFilePath")

webcamera = cv2.VideoCapture(0, cv2.CAP_DSHOW)#setando a camera integrada do pc
classificador_face = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
carrega_eye = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")
time.sleep(2)

while True:
	camera, frame = webcamera.read()

	face_cinza = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	detecta = classificador_face.detectMultiScale(face_cinza)

	for(x, y, l, a) in detecta:
		cv2.rectangle(frame, (x, y), (x+l, y+a), (255, 0, 0), 2)

		local_olho = frame[y:y + a, x:x + l]
		local_olho_cinza = cv2.cvtColor(local_olho, cv2.COLOR_BGR2GRAY)
		olho = carrega_eye.detectMultiScale(local_olho_cinza, scaleFactor=1.01, minNeighbors=1, minSize=(60,60), maxSize=(60,60))

		for(ox, oy, ol, oa) in olho:
			cv2.rectangle(local_olho, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2)
	cv2.imshow("imagem webcamera", frame)

	if cv2.waitKey(1) == ord('q'):
		break

webcamera.release()
cv2.destroyAllWindows()