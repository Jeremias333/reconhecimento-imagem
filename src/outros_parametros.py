import cv2

carrega_face = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
carrega_eye = cv2.CascadeClassifier("haarcascades/haarcascade_eye.xml")

image = cv2.imread("fotos/rostoolhos.jpg")

imagecinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = carrega_face.detectMultiScale(imagecinza)


for(x, y, l, a) in faces:
	leitura = cv2.rectangle(image, (x, y), (x + l, y + a), (255, 0, 255), 2)
	local_olho = leitura[y:y + a, x:x + l]

	local_olho_cinza = cv2.cvtColor(local_olho, cv2.COLOR_BGR2GRAY)

	olho = carrega_eye.detectMultiScale(local_olho_cinza, scaleFactor=1.05, minNeighbors=1, minSize=(150,150))

	for(ox, oy, ol, oa) in olho:
		cv2.rectangle(local_olho, (ox, oy), (ox + ol, oy + oa), (0, 0, 255), 2)

cv2.imshow("Faces", image)
cv2.waitKey()