import sys
import cv2
import numpy as np
import pandas as pd

image_path = sys.argv[1]
output_path = sys.argv[2]

points = []
img = cv2.imread(image_path)

def mouse_click(event, x, y, flags, param):
    if (event == cv2.EVENT_LBUTTONDOWN):
        cv2.circle(img, (x,y), 7, (255,255,255), 2)
        points.append([x,y])

cv2.namedWindow('my_window')
cv2.setMouseCallback('my_window', mouse_click)

while(1):
    cv2.imshow('my_window', img)
    if (cv2.waitKey(20) & 0xFF == 27): # 0x27 is the ESC button
        break

cv2.destroyAllWindows()

# convert points to pandas data frame then write to csv
df = pd.DataFrame(points)
df.to_csv(output_path, index=False, header=False)

# write modified image
cv2.imwrite('annotated-image.png', img)
