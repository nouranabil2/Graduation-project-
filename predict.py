import numpy as np
import time
import cv2

  
	
	
	

class yolo_model:
	INPUT_FILE='data/dog.jpg'
	OUTPUT_FILE='predicted.jpg'
	LABELS_FILE='data/coco.names'
	CONFIG_FILE='cfg/yolov3.cfg'
	WEIGHTS_FILE='yolov3.weights'
	CONFIDENCE_THRESHOLD=0.3
	LABELS = open(LABELS_FILE).read().strip().split("\n")

	def __init__(self):
		np.random.seed(4)
		self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
			dtype="uint8")
		self.net = cv2.dnn.readNetFromDarknet(self.CONFIG_FILE,self.WEIGHTS_FILE)
		self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
		self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
	def predict(self,image):
		(H, W) = image.shape[:2]

		# determine only the *output* layer names that we need from YOLO
		ln = self.net.getLayerNames()
		print(self.net.getUnconnectedOutLayers())
		ln = [ln[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]


		blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
			swapRB=True, crop=False)
		self.net.setInput(blob)
		start = time.time()
		layerOutputs = self.net.forward(ln)
		end = time.time()


		print("[INFO] YOLO took {:.6f} seconds".format(end - start))


		# initialize our lists of detected bounding boxes, confidences, and
		# class IDs, respectively
		boxes = []
		confidences = []
		classIDs = []

		# loop over each of the layer outputs
		
		for output in layerOutputs:
			# loop over each of the detections
			
			for detection in output:
				# extract the class ID and confidence (i.e., probability) of
				# the current object detection
				scores = detection[5:]
				classID = np.argmax(scores)
				confidence = scores[classID]
				# filter out weak predictions by ensuring the detected
				# probability is greater than the minimum probability
				if confidence > self.CONFIDENCE_THRESHOLD:
					# scale the bounding box coordinates back relative to the
					# size of the image, keeping in mind that YOLO actually
					# returns the center (x, y)-coordinates of the bounding
					# box followed by the boxes' width and height
					box = detection[0:4] * np.array([W, H, W, H])
					(centerX, centerY, width, height) = box.astype("int")

					# use the center (x, y)-coordinates to derive the top and
					# and left corner of the bounding box
					x = int(centerX - (width / 2))
					y = int(centerY - (height / 2))
					# update our list of bounding box coordinates, confidences,
					# and class IDs
					boxes.append([x, y, int(width), int(height)])
					confidences.append(float(confidence))
					classIDs.append(classID)

		# apply non-maxima suppression to suppress weak, overlapping bounding
		# boxes
		idxs = cv2.dnn.NMSBoxes(boxes, confidences, self.CONFIDENCE_THRESHOLD,
			self.CONFIDENCE_THRESHOLD)

		# ensure at least one detection exists
		if len(idxs) > 0:
			# loop over the indexes we are keeping
			predictions = dict()
			for i in idxs.flatten():
				# extract the bounding box coordinates
				(x, y) = (boxes[i][0], boxes[i][1])
				(w, h) = (boxes[i][2], boxes[i][3])

				color = [int(c) for c in self.COLORS[classIDs[i]]]
				if not classIDs[i] in predictions.keys():
					predictions[classIDs[i]]=list()
				predictions[classIDs[i]].append((x, y, w, h,confidences[i]))
				cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
				text = "{}: {:.4f}".format(self.LABELS[classIDs[i]], confidences[i])
				cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
					0.5, color, 2)

		# show the output image
		cv2.imwrite("example.png", image)
		# return a dict with classID as keys and value tuple(centerX,centerY,width,height,confidance)
		return predictions


yolo = yolo_model()
pred = yolo.predict(cv2.imread('data/dog.jpg'))
print(pred[16][0])
