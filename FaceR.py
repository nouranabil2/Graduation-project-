import face_recognition
import cv2
import pickle
import numpy as np
from imutils import paths
import os



def faces_locations(image, net, min_conf):
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    (h, w) = image.shape[:2]
    boxes = []
    
    for i in range(0, detections.shape[2]):
        conf = detections[0, 0, i, 2]
        if conf > min_conf:
            # compute the (x, y)-coordinates of the bounding box for the
            # object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            box = box[
                [1, 2, 3, 0]]
            boxt = tuple(box.astype("int").tolist())
            boxes.append(boxt)

    return boxes


def extractEmbeddings():
    # path to model file change it to ur path or just put model in the same dir with py file
    model_file = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
    config_file = 'deploy.prototxt'  # same here
    net = cv2.dnn.readNetFromCaffe(config_file, model_file)
    print("[INFO] Model Loaded.")

    #detection_method = "cnn"
    encodingsfile = 'encoding.pickle'


    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    imagePaths = list(paths.list_images("Dataset"))
    #print (imagePaths[0:],len(imagePaths))

    # initialize the list of known encodings and known names
    knownEncodings = []
    knownNames = []


    for (i, imagePath) in enumerate(imagePaths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(imagePaths)))
        name = imagePath.split(os.path.sep)[-2]
        # load the input image and convert it from BGR (OpenCV ordering)
        # to dlib ordering (RGB)
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = faces_locations(image, net, 0.7)
        #boxes = face_recognition.face_locations(rgb, model=detection_method)
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)

    print("[INFO] serializing encodings...")
    data = {"encodings": knownEncodings, "names": knownNames}
    f = open(encodingsfile, "wb")
    f.write(pickle.dumps(data))
    f.close()
    return





class FaceRecognition:
    
    def __init__(self,minconf=0.8):
        self.min_conf = minconf
        encodingFile = "encoding.pickle"
        self.data = pickle.loads(open(encodingFile, "rb").read())
        print("[INFO] encodings Loaded.")

        # path to model file change it to ur path or just put model in the same dir with py file
        self.model_file = 'res10_300x300_ssd_iter_140000_fp16.caffemodel'
        self.config_file = 'deploy.prototxt'  # same here
        self.net = cv2.dnn.readNetFromCaffe(self.config_file, self.model_file)
        print("[INFO] Model Loaded.")
    
    def get_faces_locations(self,image):
        return faces_locations(image,self.net,self.min_conf)

    def recognize(self,image):
        
        
        scale = 100
        width = int(image.shape[1] * scale / 100)
        height = int(image.shape[0] * scale / 100)
        image = cv2.resize(image, (width, height))
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = self.get_faces_locations(image)
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []


        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(self.data["encodings"], encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = self.data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number of
                # votes (note: in the event of an unlikely tie Python will
                # select first entry in the dictionary)
                name = max(counts, key=counts.get)

            # update the list of names
            names.append(name)
        
        return names,boxes

    def draw_boxes(self,image,names,boxes):
       
        # loop over the recognized faces
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                        0.75, (0, 255, 0), 2)
        # show the output frame
        return image



if __name__ =="__main__":
    model = FaceRecognition()
    cam = cv2.VideoCapture(0)
    extractEmbeddings()
    while True:
        ret, image = cam.read()
        names,boxes= model.recognize(image)
        model.draw_boxes(image,names,boxes)
        print(names)
        cv2.imshow("image", image) 
        if cv2.waitKey(1) == 27:  # break if press ESC key
            break
            