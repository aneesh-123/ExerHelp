import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import math

MODE = "MPI"

if MODE is "COCO":
    protoFile = "pose/coco/pose_deploy_linevec.prototxt"
    weightsFile = "pose/coco/pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = "pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = "pose/mpi/pose_iter_160000.caffemodel%0D"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

paths = [r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5004.jpg",
         r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5007.jpg",
         r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5009.jpg",
         r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5011.jpg"]

for path in paths:
    initial_time = time.time()

    frame = cv2.imread(path)
    frameCopy = np.copy(frame)
    frameCopy2 = np.copy(frame)
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    threshold = 0.1

    inWidth = 368
    inHeight = 368

    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    net.setInput(inpBlob)

    output = net.forward()
    H = output.shape[2]
    W = output.shape[3]

    # Empty list to store the detected keypoints
    points = []

    for i in range(nPoints):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else:
            points.append(None)

    # Draw Skeleton
    for pair in POSE_PAIRS:
        partA = pair[0]
        partB = pair[1]

        if points[partA] and points[partB]:
            cv2.line(frame, points[partA], points[partB], (255,0 , 0), 3)

    # Isolate only the arm
    for i in range(len(POSE_PAIRS)):
        #Only isolated to the right hand for now, use 5 & 6 for the left hand
        if(i == 2 or i == 3):
            cv2.line(frameCopy2, points[i], points[i+1], (0,255,0), 3)
            cv2.putText(frameCopy2, "{}".format(i), points[i], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        lineType=cv2.LINE_AA)
            if(i == 2):
                shoulderPoint = points[i]
            if(i == 3):
                elbowPoint = points[i]

    x_distance = abs(shoulderPoint[0]-elbowPoint[0])
    y_distance = abs(shoulderPoint[1]-elbowPoint[1])
    shoulder_angle = math.degrees(math.atan(x_distance/y_distance))

    targetPoint = (shoulderPoint[0], elbowPoint[1])
    cv2.circle(frameCopy2, targetPoint, 6, (0,0,255), -1)
    accuracyScore = round(((1- shoulder_angle/100) * 100), 2)
    cv2.putText(frameCopy2, ("Accuracy Score " + str(accuracyScore) + "%"), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    print("Accuracy Score ", accuracyScore)
    print(x_distance, y_distance)
    print(shoulder_angle)

    #Neat Display
    cv2.imshow('output window',frameCopy2)
    cv2.waitKey(1)
    time.sleep(0.5)