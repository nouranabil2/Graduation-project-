from FaceR import FaceRecognition,extractEmbeddings
import cv2


if __name__ =="__main__":
    model = FaceRecognition()
    extractEmbeddings()
    image=cv2.imread('download.jpg')
    names,boxes= model.recognize(image)
    model.draw_boxes(image,names,boxes)
    print(names) 