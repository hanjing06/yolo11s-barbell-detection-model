"""
Calculates the bar path using a custom YOLO model
used to detect barbells and finding the centroid
"""

import cv2 as cv

filepath = input("Import video file: ")
video = cv.VideoCapture(filepath)

if video.isOpened():
    print("Video file opeed successfully!")
else:
    print("Error: Could not open video file.")



def findcentroid(x1, y1, x2, y2):
    """
    Calculates the centroid (cx, cy) of a box
    """

    cx=(x1+x2)/2
    cy=(y1+y2)/2
    return cx,cy

def draw_path(centroid):
    """
    Draws a dot for the centroid
    """

    cv.circle(img, centroid, 2, (57, 255, 20), 2px)




