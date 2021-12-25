import cv2
import os 

path = "/home/slava/Source/YoloFaceCupBottle/Dataset"
 
filelist = []

for root, dirs, files in os.walk(path): 
	for file in files: 
		filelist.append(file)

for name in filelist: 
	if ".jpg" in name:
		print(name)
		image = cv2.imread(path + "/" + name)
		rimage = cv2.resize(image, (416, 416), interpolation = cv2.INTER_AREA)
		cv2.imwrite(path + "/" + name, rimage)
