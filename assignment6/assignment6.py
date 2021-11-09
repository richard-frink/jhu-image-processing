import sys
import os
import cv2
import numpy as np

# globally used data
ds1_input = []
ds2_s1_input = []
ds2_s2_input = []

color_list = [(255, 0, 255)]
from_center = False # Draw bounding box from upper left
show_cross_hair = False # Don't show the cross hair

def read_inputs():
    # dataset 1
    for x in range(2120, 2359):
        ds1_input.append(cv2.imread("WalkByShop1front" + str(x).zfill(4) + ".jpg"))
    # dataset 2 scenario 1
    for x in range(55, 82):
        ds2_s1_input.append(cv2.imread(str(x).zfill(3) + ".jpg"))
    # dataset 2 scenario 2
    for x in range(200, 246):
        ds2_s2_input.append(cv2.imread(str(x).zfill(3) + ".jpg"))


def process_images(images):
    bounding_box_list = []
    output_images = []
    points_traveled = []
    multi_tracker = cv2.MultiTracker_create()
    while True:
        # Draw a bounding box over all the objects that you want to track_type
        # Press ENTER or SPACE after you've drawn the bounding box
        bounding_box = cv2.selectROI('Multi-Object Tracker', images[0], from_center, 
        show_cross_hair) 

        # Add a bounding box
        bounding_box_list.append(bounding_box)

        # Press 'q' to start object tracking. You can press another key if you want to draw another bounding box.           
        print("Press q to begin tracking objects or press another key to draw the next bounding box")

        # Wait for keypress
        k = cv2.waitKey() & 0xFF

        # Start tracking objects if 'q' is pressed            
        if k == ord('q'):
            break

    cv2.destroyAllWindows()

    print("Tracking objects. Please wait...")

    for bbox in bounding_box_list:
        # Add tracker to the multi-object tracker
        multi_tracker.add(cv2.TrackerCSRT_create(), images[0], bbox)

    for image in images:
        # Update the location of the bounding boxes
        success, bboxes = multi_tracker.update(image)

        # Draw the bounding boxes on the video frame
        for i, bbox in enumerate(bboxes):
            point_1 = (int(bbox[0]), int(bbox[1]))
            point_2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
            cv2.rectangle(image, point_1, point_2, color_list[i], 2)
            points_traveled.append(point_1)
        
        # Write the frame to the output video file
        output_images.append(image)

    return output_images, points_traveled


def create_output(images, output_path):
    # make output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # print all output images
    for x in range(len(images)):
        cv2.imwrite(output_path + "/" + str(x) + ".jpg", images[x])


def create_path(image, path_traveled, output_path):
    # make output path if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    working_image = image.copy()
    first = True
    previous_point = None

    for point in path_traveled:
        if (not first):
            cv2.line(working_image, previous_point, point, color_list[0], 2)
        else:
            previous_point = point
    
    cv2.imwrite(output_path + "/path_traveled" + ".jpg", working_image)


def main():
    read_inputs()

    ds1_output, ds1_path = process_images(ds1_input)
    ds2_s1_output, ds2_s1_path = process_images(ds2_s1_input)
    ds2_s2_output, ds2_s2_path = process_images(ds2_s2_input)

    create_output(ds1_output, "dataset1_output")
    create_path(ds1_output, ds1_path, "dataset1_output")
    create_output(ds2_s1_output, "dataset2_scenario1_output")
    create_path(ds2_s1_output, ds2_s1_path, "dataset2_scenario1_output")
    create_output(ds2_s2_output, "dataset2_scenario2_output")
    create_path(ds2_s2_output, ds2_s2_path, "dataset2_scenario2_output")


main()