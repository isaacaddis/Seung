import cv2
import numpy as np
import argparse
import imutils
import time
from imutils.object_detection import non_max_suppression
import pytesseract

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", required = True, help = "Input")
parser.add_argument("-east","--east", type=str, required= True,help="path to EAST detector")
#parser.add_argument("-c","--c",type=str,help="Minimum confidence", default=0.5)
#parser.add_argument("-w", "--width",type=str, help="Width of resized image. Must be a multiple of 32 in order to work with EAST detector", default = 320)
#parser.add_argument("-e", "--height",type=str, help="Height of resized image. Must be a multiple of 32 in order to work with EAST detector", default = 320)

args = vars(parser.parse_args())

'''
    Image Operations
    Let's clean up some input data
'''
image = cv2.imread(args["image"])
orig = image.copy()
(H, W) = image.shape[:2]
rW = W/320
rH = H/320
image = cv2.resize(image, (320, 320))
(H, W) = image.shape[:2]
layerNames = [
            "feature_fusion/Conv_7/Sigmoid",
                "feature_fusion/concat_3"]
print("[INFO] Loading EAST detector")
text_dect = cv2.dnn.readNet(args["east"])
startTime = time.time()
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
            (123.68, 116.78, 103.94), swapRB=True, crop=False)
text_dect.setInput(blob)
(scores, geometry) = text_dect.forward(layerNames)
print("[INFO] Finished text detection in {:.6f} seconds".format(time.time()-startTime))

(rows, cols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, rows):
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	for x in range(0, cols):
		if scoresData[x] < .65:
			continue
		(offsetX, offsetY) = (x * 4.0, y * 4.0)
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])
boxes = non_max_suppression(np.array(rects), probs=confidences)
for (startX, startY, endX, endY) in boxes:
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)
    focus = orig[startY:endY, startX:endX]
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0, 255, 0), 2)
    #cv2.imshow(str(time.time()), focus)
cv2.imshow("Text Detect",orig)
cv2.waitKey(0)

