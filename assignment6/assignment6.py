import sys
import os
import cv2
import numpy as np

# globally used data
ds1_input = []
ds2_s1_input = []
ds2_s2_input = []

color_list = [(255, 0, 255)]

def read_inputs():
    for x in range(2120, 2359):


def process_images():



def create_output(images, output_path):
    # make output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # print all output images
    for x in range(len(images)):
        cv2.imwrite(output_path + str(x) + ".jpg", images[x])

def create_path(image, path_traveled):








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




# print final result image
# with the supplied images and the given code this should have 12 successful classifications
cv2.imwrite("coins_output/all_results.jpg", np.concatenate(horizontally_stacked_results, axis=0))