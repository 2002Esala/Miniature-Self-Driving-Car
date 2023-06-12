import cv2
import numpy as np
import utilis
import datetime

curveList = []
avgVal = 10


def getLaneCurve(image, display=2):
    global imgLaneColor
    imgCopy = image.copy()
    imgResult = image.copy()

    # STEP 1 (Thresholding)
    imgThres = utilis.thresholding(image)

    # cv2.imshow('Thres Original', imgThres)

    # STEP 2 (Warping)
    hT, wT, c = image.shape
    points = utilis.valTrackbars()
    imgWarp = utilis.warpImg(imgThres, points, wT, hT)
    imgWarpPoints = utilis.drawPoints(imgCopy, points)

    # STEP 3 (Pixel Summation and Histogram Method)
    middlePoint, imgHist = utilis.getHistogram(imgWarp, display=True, minPer=0.5, region=4)
    curveAveragePoint, imgHist = utilis.getHistogram(imgWarp, display=True, minPer=0.9)
    curveRaw = curveAveragePoint - middlePoint

    # STEP 4(Optimizing Curve)
    curveList.append(curveRaw)
    if len(curveList) > avgVal:
        curveList.pop(0)
    curveAngle = int(sum(curveList) / len(curveList))

    # STEP 5(Display)
    if display != 0:
        imgInvWarp = utilis.warpImg(imgWarp, points, wT, hT, inv=True)
        imgInvWarp = cv2.cvtColor(imgInvWarp, cv2.COLOR_GRAY2BGR)
        imgInvWarp[0:hT // 3, 0:wT] = 0, 0, 0
        imgLaneColor = np.zeros_like(image)
        imgLaneColor[:] = 0, 255, 0
        imgLaneColor = cv2.bitwise_and(imgInvWarp, imgLaneColor)
        imgResult = cv2.addWeighted(imgResult, 1, imgLaneColor, 1, 0)
        midY = 450
        cv2.putText(imgResult, str(curveAngle), (wT // 2 - 80, 85), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 3)
        cv2.line(imgResult, (wT // 2, midY), (wT // 2 + (curveAngle * 3), midY), (255, 0, 255), 5)
        cv2.line(imgResult, ((wT // 2 + (curveAngle * 3)), midY - 25), (wT // 2 + (curveAngle * 3), midY + 25),
                 (0, 255, 0), 5)
        for x in range(-30, 30):
            w = wT // 20
            cv2.line(imgResult, (w * x + int(curveAngle // 50), midY - 10),
                     (w * x + int(curveAngle // 50), midY + 10), (0, 0, 255), 2)
        # fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
        # cv2.putText(imgResult, 'FPS ' + str(int(fps)), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (230, 50, 50), 3);
    if display == 2:
        imgStacked = utilis.stackImages(0.7, ([image, imgWarpPoints, imgWarp],
                                              [imgHist, imgLaneColor, imgResult]))

        cv2.imshow('ImageStack', imgStacked)
    elif display == 1:
        cv2.imshow('Result', imgResult)

    return curveAngle


# Define a function to write a number to a log file
def write_to_log(number, filename):
    with open(filename, "a") as log_file:
        log_file.write("{} - {}\n".format(datetime.datetime.now(), number))


if __name__ == '__main__':
    cap = cv2.VideoCapture('RoadVideos/vid3.mp4')
    initialTrackbarVals = [17, 219, 56, 176]
    utilis.initializeTrackbars(initialTrackbarVals)
    frameCounter = 0
    i = 0
    while True:

        frameCounter += 1
        if cap.get(cv2.CAP_PROP_FRAME_COUNT) == frameCounter:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frameCounter = 0

        success, img = cap.read()
        img = cv2.resize(img, (480, 240))
        curve = getLaneCurve(img, 2)

        # cv2.imshow('vid2', img)
        cv2.waitKey(1)
