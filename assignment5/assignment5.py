import sys
import os
import cv2
import numpy as np
import pandas as pd

def getPseudoMask(image):
    # use hough edge detection -- this is a version of cany edge detection specifically for circles
    detected_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 200, param2 = 60, minRadius = 10, maxRadius = 100)
    # if there are circles detected then draw them
    if detected_circles is not None:
        # set the image to solid black first
        image[:] = 0
        # convert the circle parameters a, b and r to integers.
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # draw white filled circle
            cv2.circle(image, (a, b), r, (255, 255, 255), thickness=cv2.FILLED)
    return image

def getCoinCount(image):
    # use hough edge detection -- this is a version of cany edge detection specifically for circles
    detected_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 200, param2 = 75, minRadius = 10, maxRadius = 100)
    count = 0
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            count += 1
    # return total
    return count

def getCoinCountsByType(image, color_image):
    # use hough edge detection -- this is a version of cany edge detection specifically for circles
    detected_circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 20, param1 = 200, param2 = 75, minRadius = 10, maxRadius = 100)
    pennies = 0
    nickels = 0
    dimes = 0
    quarters = 0
    if detected_circles is not None:
        detected_circles = np.uint16(np.around(detected_circles))
        for pt in detected_circles[0, :]:
            a, b, r = pt[0], pt[1], pt[2]
            # default black
            color = (0,0,0)
            # check coin by radius --- not really sure how else to do this and I am not spending anymore time for the sake of my own sanity
            if (r <= 60): # if dime
                color = (10, 100, 245)
                dimes += 1
            if (r > 60 and r <=64): # if penny
                color = (222, 26, 4)
                pennies += 1
            if (r > 64 and r <= 68): # if nickel
                color = (17, 245, 55)
                nickels += 1
            if (r > 68): # if quarter
                color = (188, 44, 245)
                quarters += 1
            # draw a circle for the respective coin color
            cv2.circle(color_image, (a, b), r, color, 3)
    # return all counts and new image
    return pennies, nickels, dimes, quarters, color_image

def printCoinCounts(image, color_image, id):
    pennies, nickels, dimes, quarters, image = getCoinCountsByType(image, color_image)
    text = " pennies"
    if (pennies == 1):
        text = " penny"
    print("image " + str(id) + " has " + str(pennies) + text)
    text = " nickels"
    if (nickels == 1):
        text = " nickel"
    print("image " + str(id) + " has " + str(nickels) + text)
    text = " dimes"
    if (dimes == 1):
        text = " dime"
    print("image " + str(id) + " has " + str(dimes) + text)
    text = " quarters"
    if (quarters == 1):
        text = " quarter"
    print("image " + str(id) + " has " + str(quarters) + text)
    return image


# array of base images to be processed
input_coin_images = []
# array of image masks for the coins
coin_masks = []
# array of classified images
coin_classifications = []
# array of stacked result images
horizontally_stacked_results = []

# main loop that processes all images
for x in range(22):
    # read the images
    filepath = "coins/image_" + str(x).zfill(2) + ".jpg"
    base_image = cv2.imread(filepath)
    working_image = base_image.copy()

    # retain input image
    input_coin_images.append(working_image.copy())

    # read it again but in grayscale
    working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)

    # normalize the image - this helps for some of the pictures that are very gray or very bright
    working_image = cv2.normalize(working_image,  working_image, 0, 255, cv2.NORM_MINMAX)

    # blur the image
    working_image_blurred = cv2.blur(working_image, (4, 4))

    # get the masked image and add to our list
    working_mask = getPseudoMask(working_image_blurred.copy())
    working_mask = cv2.cvtColor(working_mask, cv2.COLOR_GRAY2RGB)
    coin_masks.append(working_mask)

    # get the coins count and classified image
    classified_image = printCoinCounts(working_image_blurred.copy(), base_image.copy(), x)
    coin_classifications.append(classified_image.copy())

    # stack results horizontally
    horizontally_stacked_results.append(np.concatenate((input_coin_images[x], coin_masks[x], coin_classifications[x]), axis=1))


# make output path if it doesn't exist
if not os.path.exists("coins_output"):
    os.makedirs("coins_output")

# print all output images
for x in range(22):
    filepath = "coins_output/coin_mask_" + str(x).zfill(2) + ".png"
    cv2.imwrite(filepath, coin_masks[x])
    filepath = "coins_output/coin_classification_" + str(x).zfill(2) + ".png"
    cv2.imwrite(filepath, coin_classifications[x])

# print final result image
# with the supplied images and the given code this should have 12 successful classifications
cv2.imwrite("coins_output/all_results.jpg", np.concatenate(horizontally_stacked_results, axis=0))