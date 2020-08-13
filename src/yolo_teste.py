import numpy as np
import cv2
import time

#criará arquivo com objetos detectados.
from csv import DictWriter

camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)#setando a camera integrada do pc
time.sleep(2)

#variaveis de captura
h, w = None, None

#carrega os arquivos com o nome dos objetos que foi treinado para identificar
with open("OpenCVYolo/yoloDados/YoloNames.names") as f:
	#cria uma lista com todos os nomes
	labels = [line.strip() for line in f]

#carrega arquivos treinados pelo framework
network = cv2.dnn.readNetFromDarknet("OpenCVYolo/yoloDados/yolov3.cfg", "OpenCVYolo/yoloDados/yolov3.weights")

#captura ua lista com todos os nomes dos objetos treinados pelo framework
layers_names_all = network.getLayerNames()

#obtendo apenas o nome de camadas de saida que precisamos para o algoritmo Yolov3
#com função de retornar o indice das camadas com saidas desconectadas

layers_names_output = \
	[layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Definir probabilidade minima para eliminar previsões fracas
probability_minimum = 0.5

#Definir limite para filtrar caixas delimitadoras fracas
#com supressão não máxima
threshold = 0.3

#Gera cores aleatórias nas caixas de cada objeto detectados.
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

#loop de captura e detecção de objetos
with open("teste.csv", "w") as arquivo:
	cabecalho = ["Detectado", "Acuracia"]
	escritor_csv = DictWriter(arquivo, fieldnames=cabecalho)
	escritor_csv.writeheader()

	while True:
		#captura de camera frame por frame
		_, frame = camera.read()

		if w is None or h is None:
			#fatiar apenas dois primeiros elementos da tupla
			h, w = frame.shape[:2]

		#A forma resultante possui um numero de quadros, numero de canais, largura e altura
		#E.G.:
		blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)

		#Implementando o passe direto com nosso blob somente atraves das camadas de saída
		#Calculo ao mesmo tempo, tempo necessário para encaminhamento
		network.setInput(blob) #definindo blob como entrada para a rede
		start = time.time()
		output_from_network = network.forward(layers_names_output)
		end = time.time()

		#mostrando o tempo gasto para um unico quadro atual
		print("tempo gasto atual {:.5f} segundos".format(end - start))

		#preparando listas para caixas delimitadoras detectadas

		bounding_boxes = []
		confidences = []
		class_numbers = []

		#passando por todas as camadas de saída após o avanço da alimentação
		#fase de detecção dos objetos

		for result in output_from_network:
			for detected_objects in result:
				scores = detected_objects[5:]
				class_current = np.argmax(scores)

				confidence_current = scores[class_current]

				#eliminando previsões fracas com probablilidade minima
				if confidence_current > probability_minimum:
					box_current = detected_objects[0:4] * np.array([w, h, w, h])
					x_center, y_center, box_width, box_height = box_current
					x_min = int(x_center - (box_width / 2))
					y_min = int(y_center - (box_height / 2))

					#Adicionando resultados em listas preparadas
					bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
					confidences.append(float(confidence_current))
					class_numbers.append(class_current)

		results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, probability_minimum, threshold)

		#verificando se existe pelo menos um objeto detectado
		if len(results) > 0:
			for i in results.flatten():
				x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
				box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
				colours_box_current = colours[class_numbers[i]].tolist()
				cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height), colours_box_current, 2)


				#Preparando texto com rótulo e acuracia para o objeto detectado.
				text_box_current = "{}: {:.4f}".format(labels[int(class_numbers[i])], confidences[i])

				# Coloca o texto nos objetos detectados
				cv2.putText(frame, text_box_current, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colours_box_current, 2)

				escritor_csv.writerow( {"Detectado": text_box_current.split(":")[0], "Acuracia": text_box_current.split(":")[1]})
				print(text_box_current.split(":")[0] +" - "+ text_box_current.split(":")[1])


		cv2.namedWindow('Yolo v3 WebCamera', cv2.WINDOW_NORMAL)
		cv2.imshow("Yolo v3 Cam", frame)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break


camera.release()
cv2.destroyAllWindows()


