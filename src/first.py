import cv2

carrega_algoritmo = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")

image = cv2.imread("fotos/eu.jpg")

imagecinza = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = carrega_algoritmo.detectMultiScale(imagecinza)

print(faces)

for(x, y, l, a) in faces:
	cv2.rectangle(image, (x, y), (x + l, y + a), (0,255,0), 2)

cv2.imshow("Faces", image)
cv2.waitKey()