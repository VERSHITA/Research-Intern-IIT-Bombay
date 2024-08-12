import argparse
import cv2
import numpy as np
import math
import csv
import gc
import matplotlib.pyplot as plt

def right_left(anglet):
    if 0 <= anglet <= 90 or 270 <= anglet <= 360:
        return 1, 0
    else:
        return 0, 1

def angle_tanfunction(x, y):
    if x > 0 and y >= 0:
        return math.degrees(math.atan(y / x))
    elif x < 0 and y >= 0:
        return 180 + math.degrees(math.atan(y / x))
    elif x < 0 and y < 0:
        return 180 + math.degrees(math.atan(y / x))
    elif x > 0 and y < 0:
        return 360 + math.degrees(math.atan(y / x))
    elif x == 0 and y > 0:
        return 90
    elif x == 0 and y < 0:
        return 270
    return 0

def process_batch(start, end, csv_writer):
    for i in range(start, end):
        path = f"C:/Users/vershi/OneDrive/Desktop/slablab/circle_bot/v_20_vacc_0/frame/{i}.jpg"
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            print(f"Image {path} not found or cannot be opened.")
            continue  # Skip to the next iteration if the image is not loaded

        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Set up the SimpleBlobDetector parameters
        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 165
        params.maxArea = 551

        # Create a SimpleBlobDetector
        detector = cv2.SimpleBlobDetector_create(params)

        # Detect blobs
        keypoints = detector.detect(gray)

        # Extract blob information
        detected_circles = [(float(k.pt[0]), float(k.pt[1]), float(k.size / 2)) for k in keypoints]

        # Draw detected blobs using Matplotlib
       
            # Draw arrow between two points (example using the first two detected circles)
        if len(detected_circles) >= 2:
            if detected_circles[0][2] > detected_circles[1][2]:
                start_point = (detected_circles[0][0], detected_circles[0][1])
                end_point = (detected_circles[1][0], detected_circles[1][1])
            else:
                start_point = (detected_circles[1][0], detected_circles[1][1])
                end_point = (detected_circles[0][0], detected_circles[0][1])

            center_x = (detected_circles[0][0] + detected_circles[1][0]) / 2
            center_y = (detected_circles[0][1] + detected_circles[1][1]) / 2

            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]

               

            anglet1 = angle_tanfunction(dx, dy)
            anglet = round(anglet1, 3)
            right, left = right_left(anglet)

            list_row = [center_x * 0.316, center_y * 0.316, anglet, right, left]
            csv_writer.writerow(list_row)  # Write the data to CSV

                

        del img, gray, detected_circles, keypoints
        gc.collect()
        print("Processed image:", i)

def main():
    parser = argparse.ArgumentParser(description="Process images in batches.")
    parser.add_argument("--start", type=int, required=True, help="Starting index of images to process.")
    parser.add_argument("--end", type=int, required=True, help="Ending index of images to process.")
    parser.add_argument("--output", type=str, default="point0.csv", help="Output CSV file name.")
    
    args = parser.parse_args()
    
    # Open the CSV file in write mode
    file = open(args.output, "a+", newline='')
    csv_writer = csv.writer(file)
    csv_writer.writerow(["Center X", "Center Y", "Orientation", "Right", "Left"])

    try:
        process_batch(args.start, args.end, csv_writer)
    finally:
        file.close()  # Ensure the CSV file is properly closed

if __name__ == "__main__":
    main()


#python process_images.py --start 0 --end 1000 --output point1.csv