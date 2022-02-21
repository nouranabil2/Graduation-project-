from FaceR import FaceRecognition,extractEmbeddings
import cv2


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