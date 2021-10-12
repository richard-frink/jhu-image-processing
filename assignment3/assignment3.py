import sys
import cv2
import numpy as np
import pandas as pd
from common import Sketcher

img = cv2.imread("pumpkins.jpg")
masking_img = img.copy()
img_to_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
final_color_masked = cv2.cvtColor(img_to_gray, cv2.COLOR_GRAY2RGB)

# draw the white parts to create the mask
#   just runs during the while is True
mark = np.zeros(img.shape[:2], np.uint8)
sketch = Sketcher('img', [masking_img, mark], lambda : ((255, 255, 255), 255))

while True:
    ch = cv2.waitKey()
    # break on the ESC key
    if ch == 27:
        break
    # calculate and show the masked image
    if ch == ord('r'):
        # find all marked portions of the image (where fully white)
        # take the grayscale image and for every part that was marked
        #   replace gray with the original "img" color
        masking_img
        for r in range(masking_img.shape[0]):
            for c in range(masking_img.shape[1]):
                if (np.array_equal(masking_img[r,c],(255,255,255))):
                    final_color_masked[r,c] = img[r,c]
        sketch.show()
        cv2.imwrite('masking_image.png', masking_img)

cv2.destroyAllWindows()

# write modified image
cv2.imwrite('grayscale_with_color_mask.png', final_color_masked)
