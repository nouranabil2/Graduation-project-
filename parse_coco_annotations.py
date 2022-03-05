import readline
import cv2
import numpy as np
import os
import json
import sys
import pickle
from os import listdir
from os.path import isfile, join
VAL_ANNOTATION_PATH = "data/annotations/instances_val2017.json"
TRAIN_ANNOTATION_PATH = "data/annotations/instances_train2017.json"
TRAIN_OUTPUT_PATH = "data/dataset/train.pk1"
VAL_OUTPUT_PATH = "data/dataset/val.pk1"

COCO_CLASSES = "data/coco.names"
TRAIN_IMAGES_PATH = "data/train2017"
VAL_IMAGES_PATH ="data/val2017"

TRAIN_CONVERTED_COCO_ANNOTATIONS_PATH = "data/dataset/train2017.txt"
VAL_CONVERTED_COCO_ANNOTATIONS_PATH = "data/dataset/val2017.txt"




class COCO:
    def parse(self,json_annotation_path):
        try:
            json_data = json.load(open(json_annotation_path))
            images_info = json_data["images"]
            cat_info = json_data["categories"]
            data = {}
            length = len(json_data["annotations"])
            percent_pro = 0
            for annotation in json_data["annotations"]:
                image_id = annotation["image_id"]
                category_id = annotation["category_id"]

                file_name=None
                img_width=None
                img_height=None
                img_class=None

                for info in images_info:
                    if info["id"] == image_id:
                        file_name,img_width,img_height = info["file_name"].split(".")[0],info["width"],info["height"]
                        break
                for category in cat_info:
                    if category_id == category["id"]:
                        img_class = category["name"]
                        break
                image_size = {
                    "width":img_width,
                    "height":img_height,
                    "depth":"3"
                }
                bounding_box={
                    "xmin": annotation["bbox"][0],
                    "ymin": annotation["bbox"][1],
                    "xmax": annotation["bbox"][2] + annotation["bbox"][0],
                    "ymax": annotation["bbox"][3] + annotation["bbox"][1]
                }
                obj_info={
                    "name":img_class,
                    "bndbox":bounding_box
                }
                if file_name not in data:
                    obj = {
                        "num_obj":"1",
                        "0":obj_info
                    }
                    data[file_name] = {
                        "size":image_size,
                        "objects":obj
                    }
                elif file_name in data:
                    obj_idx = data[file_name]["objects"]["num_obj"]
                    data[file_name]["objects"][obj_idx]=obj_info
                    data[file_name]["objects"]["num_obj"]=str(int(data[file_name]["objects"]["num_obj"]) + 1)
                percent = (float(percent_pro) / float(length)) * 100
                print(str(percent_pro) + "/" + str(length) + " total: " + str(round(percent, 2)))
                percent_pro += 1
            #print(json.dumps(data, indent=4, sort_keys = True))
            return True, data            
        except Exception as e:
            msg = str(e)
            print(msg)
            return False, msg
    def convert_coco_annotation(self,output_path,data,is_train=True):
        replace_dict = {"couch": "sofa", "airplane": "aeroplane", "tv": "tvmonitor", "motorcycle": "motorbike"}
        classes = [line.strip() for line in open(COCO_CLASSES).readlines()]
        if os.path.exists(output_path):
            os.remove(output_path)
        if is_train:
            data_path = TRAIN_IMAGES_PATH
        else:
            data_path =VAL_IMAGES_PATH
        images_path = [image for image in listdir(data_path) if isfile(join(data_path,image))]
        count=0
        with open(output_path,"a") as f:
            for image_path in images_path:
                image_idx = image_path.split(".")[0]
                annotation = os.path.join(data_path,image_path)
                if image_idx in data:
                    objects = data[image_idx]["objects"]
                    for key,value in objects.items():
                        if key == 'num_obj': continue
                        if value["name"] not in classes:
                            print(value["name"])
                            class_name = replace_dict[value["name"]]
                            class_idx = classes.index(class_name)
                        else:
                            class_idx = classes.index(value["name"])
                        xmin = int(value["bndbox"]["xmin"])
                        ymin = int(value["bndbox"]["ymin"])
                        xmax = int(value["bndbox"]["xmax"])
                        ymax = int(value["bndbox"]["ymax"])
                        annotation += " " +",".join([str(xmin), str(ymin), str(xmax), str(ymax), str(class_idx)])
                else:
                    continue
                f.write(annotation+"\n")
                count+=1
        print(count)
                            
if __name__ == "__main__":

    coco = COCO()
    print("parsing val data ...")
    data = coco.parse(VAL_ANNOTATION_PATH) 
    coco.convert_coco_annotation(VAL_CONVERTED_COCO_ANNOTATIONS_PATH,data[1],False)
    with open(VAL_OUTPUT_PATH,"wb") as f:
        pickle.dump(data[1],f,protocol=pickle.HIGHEST_PROTOCOL) 
    

    print("parsing training data ...")
    data = coco.parse(TRAIN_ANNOTATION_PATH) 
    with open(TRAIN_OUTPUT_PATH,"wb") as f:
        pickle.dump(data[1],f,protocol=pickle.HIGHEST_PROTOCOL) 
    coco.convert_coco_annotation(TRAIN_CONVERTED_COCO_ANNOTATIONS_PATH,data[1])



