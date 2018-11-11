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
(H, W) = image.shape[:2]
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.GaussianBlur(image,(5,5),5)
'''
if args["width"] and args["height"] is not None:
    (newW, newH) = (args["width"], args["height"])
    rW = W/float(newW)
    rH = H/float(newH)
    image = cv2.resize(image, (newW, newH))
else:
    (newW, newH) = (320,320)
    rW = W/float(newW)
    rH = H/float(newH)
'''
image = cv2.resize(image, (320, 320))
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
print("[INFO] Finished text detection in {::.6f} seconds".format(time.time()-startTime))
'''
(numRows, numCols) = scores.shape[2:4]
rects = []
confidences = []

for y in range(0, numRows):
    scoresData = scores[0,0,y]
    xData0 = geometry[0,0,y]
    xData1 = geometry[0,1,y]
    xData2 = geometry[0,2,y]
    xData3 = geometry[0,3,y]
    anglesData = geometry[0,4,y]
'''

