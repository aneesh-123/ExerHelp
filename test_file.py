import cv2
import time
import numpy as np
import matplotlib.pyplot as plt
import math

def find_closest_index(desired_milliseconds, times):
    difference = 100000000
    index = 0
    for i in range(len(times)):
        test_difference = abs(desired_milliseconds-times[i][1])
        if(test_difference < difference):
            difference = test_difference
            index = i
    return index


cameraCapture = cv2.VideoCapture(r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5021.MOV")
success, frame = cameraCapture.read()
i =0

total_time = 0
times = []
while success:
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow('Test camera', frame)
    success, frame = cameraCapture.read()
    milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
    times.append([i,milliseconds])

    if(int(milliseconds) > total_time):
        total_time = int(milliseconds)

    print(i, int(milliseconds))
    i = i +1


cv2.destroyAllWindows()
cameraCapture.release()

########################################### PASS SOME FRAMES OF THE VIDEO INTO THE NEURAL NETWORK ###########################

timestamp = 0

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

print("here")
print(timestamp, total_time)
while(timestamp < total_time):
    start_time = time.time()
    cap = cv2.VideoCapture(r"C:\Users\Aneesh\Pictures\MenloPics\IMG_5021.MOV")
    cap.set(1, find_closest_index(timestamp, times))
    ret, test = cap.read()

    if ret:
        cv2.imwrite("image"+str(timestamp)+".jpg", test)

    frame = cv2.imread("image"+str(timestamp)+".jpg")

    frame = cv2.imread("image" + str(timestamp) + ".jpg")
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
    print(points)


cap.release()
cv2.destroyAllWindows()