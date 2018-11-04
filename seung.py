import cv2
import numpy as np
import argparse
import imutils

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", required = True, help = "Input")
args = vars(parser.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, height = 500)
