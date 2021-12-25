import cv2
import numpy as np
import time

#params
#weights = "yolo-obj_last.weights"
weights = "3yolo-obj_last.weights"
#weights = "yolo-obj_final.weights"
#cfg = "yolo-obj.cfg"
cfg = "yolov3.cfg"
im_size = 416
conff = 0.1
video = "outpy.avi"

control_distance = int(0.15*im_size)

#Load YOLO
net = cv2.dnn.readNet(weights, cfg)
classes = ["Face", "Cup", "Bottle"]

layer_names = net.getLayerNames()
outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors= np.random.uniform(0,255,size=(len(classes),3))

#Loading image
cap=cv2.VideoCapture(video) #0 for 1st webcam
font = cv2.FONT_HERSHEY_PLAIN
starting_time= time.time()
frame_id = 0

def is_intersection(face_boxes, cup_boxes, bottle_boxes):
    for i in range(len(face_boxes)):
        for j in range(len(cup_boxes)):
            xf, yf = face_boxes[i]
            xc, yc = cup_boxes[j]
            distance = int(((xf-xc)**2+(yf-yc)**2)**(0.5))
            if distance > control_distance:
                return True

    for i in range(len(face_boxes)):
        for j in range(len(cup_boxes)):
            xf, yf = face_boxes[i]
            xb, yb = bottle_boxes[j]
            distance = int(((xf-xb)**2+(yf-yb)**2)**(0.5))
            if distance > control_distance:
                return True

    return False

while True:
    ret, frame= cap.read() # 
    frame_id+=1

    # if !ret:
    #     break

    height,width,channels = frame.shape
    #detecting objects
    blob = cv2.dnn.blobFromImage(frame,0.00392,(im_size,im_size),(0,0,0),True,crop=False) #reduce 416 to 320   

    net.setInput(blob)
    outs = net.forward(outputlayers)
    #print(outs[1])

    #Showing info on screen/ get confidence score of algorithm in detecting an object in blob
    class_ids=[]
    confidences=[]
    boxes=[]

    face_boxes=[]
    cup_boxes=[]
    bottle_boxes=[]

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print(confidence)
            if confidence > conff:
                print(confidence)
                #onject detected
                center_x= int(detection[0]*width)
                center_y= int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                #rectangle co-ordinaters
                x=int(center_x - w/2)
                y=int(center_y - h/2)
                #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                boxes.append([x,y,w,h]) #put all rectangle areas
                confidences.append(float(confidence)) #how confidence was that object detected and show that percentage
                class_ids.append(class_id) #name of the object tha was detected

                if class_id == 0:
                    face_boxes.append([center_x, center_y])
                elif class_id == 1:
                    cup_boxes.append([center_x, center_y])
                elif class_id == 2:
                    bottle_boxes.append([center_x, center_y])

    indexes = cv2.dnn.NMSBoxes(boxes,confidences,0.4,0.6)

    for i in range(len(boxes)):
        if i in indexes:
            x,y,w,h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence= confidences[i]
            color = colors[class_ids[i]]
            cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)
            cv2.putText(frame,label+" "+str(round(confidence,2)),(x,y+30),font,1,(255,255,255),2)

    elapsed_time = time.time() - starting_time
    fps=frame_id/elapsed_time
    cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)

    #calculating here

    success = is_intersection(face_boxes, cup_boxes, bottle_boxes)

    if success == True:
        cv2.putText(frame,"DRINK",(10,100),font,2,(0,0,255),1)
    else:
        cv2.putText(frame,"NOT DRINK",(10,100),font,2,(0,255,0),1)

    cv2.imshow("Image",frame)
    key = cv2.waitKey(1) #wait 1ms the loop will start again and we will process the next frame
    
    if key == 27: #esc key stops the process
        break;
    #cv2.waitKey(0)
    
cap.release()    
cv2.destroyAllWindows()