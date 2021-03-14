import cv2

def find_closest_index(desired_milliseconds, times):
    difference = 100000000
    index = 0
    for i in range(len(times)):
        test_difference = abs(desired_milliseconds-times[i][1])
        if(test_difference < difference):
            difference = test_difference
            index = i
    return index

cameraCapture = cv2.VideoCapture(r"C:\Users\Aneesh\Pictures\MenloPics\video0.mov")

success, frame = cameraCapture.read()
i =0

times = []
while success:
    if cv2.waitKey(1) == 27:
        break
    cv2.imshow('Test camera', frame)
    success, frame = cameraCapture.read()
    milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
    times.append([i,milliseconds])
    '''seconds = milliseconds//1000
    milliseconds = milliseconds%1000
    minutes = 0
    hours = 0
    if seconds >= 60:
        minutes = seconds//60
        seconds = seconds % 60

    if minutes >= 60:
        hours = minutes//60
        minutes = minutes % 60'''

    print(i, int(milliseconds))
    i = i +1

cv2.destroyAllWindows()
cameraCapture.release()


cap = cv2.VideoCapture(r"C:\Users\Aneesh\Pictures\MenloPics\video0.mov")

cap.set(1, find_closest_index(12590, times))
ret, frame = cap.read()
cv2.imshow('frame', frame)
cv2.waitKey()

cap.release()
cv2.destroyAllWindows()
