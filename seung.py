import cv2
import numpy as np
import argparse
import imutils
import pytesseract

parser = argparse.ArgumentParser()
parser.add_argument("-i","--image", required = True, help = "Input")
parser.add_argument("-east","--east", type=str, required= True,help="path to EAST detector")
parser.add_argument("-c","--c",type=str,help="Minimum confidence", default=0.5)
parser.add_argument("-w", "--width",type=str, help="Width of resized image. Must be a multiple of 32 in order to work with EAST detector", default = 320)
parser.add_argument("-e", "--height",type=str, help="Height of resized image. Must be a multiple of 32 in order to work with EAST detector", default = 320)

args = vars(parser.parse_args())

image = cv2.imread(args["image"])
image = imutils.resize(image, height = 500)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(5,5),5)
