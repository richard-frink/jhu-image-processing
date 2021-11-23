import sys
import os
import cv2 as cv
import numpy as np


def read_input(path):
    return cv.imread("dataset/" + str(path),0)


def create_csv_output(csv_lines, output_path):
    final_content = ""
    for line in csv_lines:
        final_content = str(final_content) + str(line) + "\n"
    # write csv
    csv = open(output_path, "w")
    csv.write(final_content)


def calculate_image_features(path):
    image = read_input(path)
    comma = ","
    csv_result = str(path)

    # ORB setup
    orb = cv.ORB_create(nfeatures=30)
    kp = orb.detect(image,None)
    kp, des = orb.compute(image, kp)

    count = 1
    descCount = 0
    for description_array in des:
        for description in description_array:
            descCount += 1
            csv_result = str(csv_result) + str(comma) + str(description)
        if count == 25: # cut it short, we need all of our objects to generate 30 points
            return csv_result
        count += 1
    
    # we need to make sure every image we analyze has the exact same number of descriptions
    #   so we pad the ones that are lacking with 0s
    while descCount < 800:
        descCount += 1
        csv_result = str(csv_result) + str(comma) + str(0)
    return csv_result


def calculate_all_image_features():
    results = []
    for x in range(0,113):
        path = "image_" + str(x) + ".jpg"
        results.append(calculate_image_features(path))
    create_csv_output(results, "all_features.csv")
    

def main():
    args = None
    # read the command line args
    if len(sys.argv) == 1:
        print("Calculating features for all images")
        calculate_all_image_features()
    else:
        args =  sys.argv[1]
        result = calculate_image_features(args)
        print(result)


main()