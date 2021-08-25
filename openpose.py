import cv2


BODY_PARTS = { "Head": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
                "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
                "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "Chest": 14,
                "Background": 15 }

POSE_PAIRS = [ ["Head", "Neck"], ["Neck", "RShoulder"], ["RShoulder", "RElbow"],
                ["RElbow", "RWrist"], ["Neck", "LShoulder"], ["LShoulder", "LElbow"],
                ["LElbow", "LWrist"], ["Neck", "Chest"], ["Chest", "RHip"], ["RHip", "RKnee"],
                ["RKnee", "RAnkle"], ["Chest", "LHip"], ["LHip", "LKnee"], ["LKnee", "LAnkle"] ]
    

protoFile = "/data0/yugyoung/VTON/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "/data0/yugyoung/VTON/pose_iter_160000.caffemodel"
 

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)


image = cv2.imread("/data0/yugyoung/VTON/02_4_full.jpg")

imageHeight, imageWidth, _ = image.shape
 

inpBlob = cv2.dnn.blobFromImage(image, 1.0 / 255, (imageWidth, imageHeight), (0, 0, 0), swapRB=False, crop=False)
 

net.setInput(inpBlob)


output = net.forward()


H = output.shape[2]
W = output.shape[3]
print("Image ID : ", len(output[0]), ", H : ", output.shape[2], ", W : ",output.shape[3])


points = []
for i in range(0,15):

    probMap = output[0, i, :, :]
 
    
    minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

    
    x = (imageWidth * point[0]) / W
    y = (imageHeight * point[1]) / H

    
    if prob > 0.1 :    
        cv2.circle(image, (int(x), int(y)), 3, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
        #cv2.putText(image, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, lineType=cv2.LINE_AA)
        points.append((int(x), int(y)))
    else :
        points.append(None)

#cv2.imshow("Output-Keypoints",image)
#cv2.waitKey(0)
cv2.imwrite('output-keypoints.jpg',image)

imageCopy = image


for pair in POSE_PAIRS:
    partA = pair[0]             
    partA = BODY_PARTS[partA]   
    partB = pair[1]             
    partB = BODY_PARTS[partB]   
    
    
    if points[partA] and points[partB]:
        cv2.line(imageCopy, points[partA], points[partB], (0, 255, 0), 2)


#cv2.imshow("Output-Keypoints",imageCopy)
#cv2.waitKey(0)
#cv2.destroyAllWindows()
cv2.imwrite("Output-Keypoints_lines.jpg",imageCopy)