import cv2 as cv
import numpy as np
import pandas as pd
# from PIL import Image, ImageDraw
# import matplotlib.pyplot as plt
from djitellopy import Tello


# Note: pattern must be a square (i.e., pattern_dim x pattern_dim),
#   but can have different number of grid units in x and y axes.


# Note: these parameters will need to change as the drone is farther from the camera, and the resolution of the video.
perimeter_thresh = 50
area_thresh = 2000

cnt_approx_factor = 0.05
pattern_width = 10  # grid units
pattern_height = 10  # grid units
pattern_dim = 400  # pixels
white_black_thresh = 100

hopfield_n = 100
hopfield_patterns = 10

used_patterns = [0] * hopfield_patterns


# Sigma sign function
def sign(x):
    if (x >= 0):
        return 1
    else:
        return -1


# Calculates for the new network with weight
def run(network, weights):
    sizeofNet = len(network)
    new_network = np.zeros(sizeofNet)

    for i in range(sizeofNet):
        h = 0
        for j in range(sizeofNet):
            h += weights[i, j] * network[j]
        new_network[i] = sign(h)
    return new_network


def init_hopfield():
    expected_patterns = []
    for RUN in range(10):
        temp = np.loadtxt("csv2/test" + str(RUN) + ".csv",
                          delimiter=",", dtype=int)
        temp = np.delete(temp, 0, 0)
        expected_patterns.append(temp)

    weights = np.zeros((hopfield_n, hopfield_n))

    for p2 in range(hopfield_patterns):
        for i in range(hopfield_n):
            for j in range(hopfield_n):
                if i != j:
                    weights[i, j] += expected_patterns[p2][i] * expected_patterns[p2][j]
    weights /= hopfield_n

    return expected_patterns, weights


def recognize_pattern(input_pattern, expected_patterns, weights, tello):
    # print(input_pattern)
    recognized_pattern = run(input_pattern, weights)

    if np.array_equal(expected_patterns[0], recognized_pattern) and used_patterns[0] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Take off")
        tello.takeoff()
        used_patterns[0] = 1
    elif np.array_equal(expected_patterns[1], recognized_pattern) and used_patterns[1] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Move Forward 60cm (~2ft)")
        tello.set_speed(10)
        tello.move_forward(60)
        used_patterns[1] = 1
    elif np.array_equal(expected_patterns[2], recognized_pattern) and used_patterns[2] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Go Over")
        tello.set_speed(10)
        tello.move_up(30)
        tello.move_forward(60)
        tello.move_down(30)
        used_patterns[2] = 1
    elif np.array_equal(expected_patterns[3], recognized_pattern) and used_patterns[3] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Turn 180 Degrees")
        tello.rotate_clockwise(180)
        used_patterns[3] = 1
    elif np.array_equal(expected_patterns[4], recognized_pattern) and used_patterns[4] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Go Around")
        tello.set_speed(10)
        tello.move_left(30)
        tello.move_forward(60)
        tello.move_right(30)
        used_patterns[4] = 1
    elif np.array_equal(expected_patterns[5], recognized_pattern) and used_patterns[5] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Flip Forward")
        tello.flip_forward()
        used_patterns[5] = 1
    elif np.array_equal(expected_patterns[9], recognized_pattern) and used_patterns[9] == 0:
        print("Battery: " + str(tello.get_battery()) + "%")
        print("Land")
        tello.land()
        used_patterns[9] = 1
    '''
    for i, pattern in enumerate(expected_patterns):
        if np.array_equal(pattern, recognized_pattern) and used_patterns[i] == 0:
            # TODO: IMPORTANT!!! Only recognize a pattern once, do not send 30 of the same command to the drone.
            used_patterns[i] = 1
            print(pattern)
    '''

def detect_shape(cnt):
    perimeter = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
    return len(approx) == 4


def sort_corners(corners):
    # Sort by x-coordinate values.
    corners.sort(key=lambda x: x[0])

    # Separate left and right corners by evaluating x-coordinates.
    left_corners = corners[:2]
    right_corners = corners[2:]

    # Separate left top and bottom corners by evaluating y-coordinates.
    top_left = left_corners[0] if left_corners[0][1] < left_corners[1][1] else left_corners[1]
    bot_left = left_corners[1] if left_corners[0][1] < left_corners[1][1] else left_corners[0]

    # Separate right top and bottom corners by evaluating y-coordinates.
    top_right = right_corners[0] if right_corners[0][1] < right_corners[1][1] else right_corners[1]
    bot_right = right_corners[1] if right_corners[0][1] < right_corners[1][1] else right_corners[0]

    return [top_left, top_right, bot_left, bot_right]


def decode_pattern(frame, cnt):
    corners = []
    for pt in cnt:
        corners.append([pt[0][0], pt[0][1]])
    # Sort corners in correct order for perspective transformation.
    corners = sort_corners(corners)
    # print(corners)

    # Apply perspective transformation, used to better segment and read each grid unit in pattern.
    orig_pts = np.float32(corners)
    result_pts = np.float32([[0, 0], [pattern_dim, 0], [0, pattern_dim], [pattern_dim, pattern_dim]])
    matrix = cv.getPerspectiveTransform(orig_pts, result_pts)
    pattern = cv.warpPerspective(frame, matrix, (400, 400))
    # cv.imshow("Original Pattern", pattern)

    decoded_pattern = pattern.copy()  # For visualization. Do not add text to pattern or it may disrupt BGR values.

    # TODO: Make this filtering better so lighting has a minimal effect.
    gray = cv.cvtColor(pattern, cv.COLOR_BGR2GRAY)
    _, pattern = cv.threshold(gray, white_black_thresh, 255, cv.THRESH_BINARY)
    # cv.imshow("Filtered Pattern", pattern2)

    # Estimate centroids of pattern grid units.
    horizontal_grid_units = pattern_width + 2  # Add two for black border grid units
    vertical_grid_units = pattern_height + 2  # Add two for black border grid units
    grid_unit_width = pattern_dim / horizontal_grid_units
    grid_unit_height = pattern_dim / vertical_grid_units

    # Decode each grid unit as 0 or 1 based on color at centroid.
    #   White -> 0, Black -> 1
    pattern_bits = []
    for i in range(1, vertical_grid_units-1):
        for j in range(1, horizontal_grid_units-1):
            cX = int(j * grid_unit_width + grid_unit_width / 2)
            cY = int(i * grid_unit_height + grid_unit_height / 2)
            if np.mean(pattern[cY, cX]) > white_black_thresh:
                # Black Grid Unit, i.e., 1.
                decoded_pattern = cv.putText(decoded_pattern, '1', (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                                             cv.LINE_AA)
                pattern_bits.append(1)
            else:
                # White Grid Unit, i.e., 0.
                decoded_pattern = cv.putText(decoded_pattern, '0', (cX, cY), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2,
                                             cv.LINE_AA)
                pattern_bits.append(-1)
            # cv.circle(pattern, (x, y), 3, (0, 0, 255), -1)

    # cv.imshow("Decoded Pattern", decoded_pattern)

    return pattern_bits


def process_frame(frame):
    copy = frame.copy()

    gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    # cv.imshow("Gray", gray)

    gray = cv.GaussianBlur(gray, (7, 7), 0)

    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    # cv.imshow("Thresh", thresh)

    edges = cv.Canny(thresh, 100, 200)
    # cv.imshow("Edges", edges)

    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Verify contours retrieved, otherwise an exception may be thrown (e.g., max() on empty sequence).
    if len(contours) <= 0:
        return

    # Record all children based on their centroids, later used to determine if the child has a parent.
    centroids = []
    for i, cnt in enumerate(contours):
        perimeter = cv.arcLength(cnt, True)
        if perimeter > perimeter_thresh:
            M = cv.moments(cnt)
            cX = 0
            cY = 0
            if M["m00"] > 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
            centroids.append([cX, cY])

    # For each valid contour (meeting perimeter/area thresholds), determine its number of children based on children
    #   centroids and parent bounding box.
    child_counts = [0] * len(contours)
    for i, cnt in enumerate(contours):
        perimeter = cv.arcLength(cnt, True)
        area = cv.contourArea(cnt)
        if perimeter > perimeter_thresh and area > area_thresh:
            x, y, w, h = cv.boundingRect(cnt)
            for centroid in centroids:
                cX = centroid[0]
                cY = centroid[1]
                if x < cX < (x + w) and y < cY < (y + h):
                    child_counts[i] += 1

    # The contour with the greatest number of children is considered the pattern.
    #   Note: Current method only supports one pattern at a time, which should be fine for our purposes.
    #   Note: AprilTags takes number of *quadrilateral* children.
    # TODO: Possible improvement, calculate child_counts based on number of corners detected -- seems more consistent
    #   than using the contours for the shapes inside of the pattern.
    max_children_count = max(child_counts)
    max_children_cnt = contours[child_counts.index(max_children_count)]

    # Verify shape is a quadrilateral; this helps with bad detections (i.e., non-patterns), but still happens at times.
    if max_children_count > 0 and detect_shape(max_children_cnt):
        # area = cv.contourArea(max_children_cnt)
        # print(area)

        # Approximate the contour to yield only four corners.
        perimeter = cv.arcLength(max_children_cnt, True)
        approx_cnt = cv.approxPolyDP(max_children_cnt, cnt_approx_factor * perimeter, True)

        # Further verification for correct detection (i.e., four corners for square pattern).
        if len(approx_cnt) == 4:
            # Draw pattern bounding box and corners.
            cv.drawContours(copy, [approx_cnt], 0, (0, 255, 0), 2)
            for pt in approx_cnt:
                x, y = pt[0]
                cv.circle(copy, (x, y), 3, (0, 0, 255), 5)

            cv.imshow('Output', copy)

            # Decode the binary representation of the pattern.
            return decode_pattern(frame, approx_cnt)


def main():
    # Connect to Tello EDU Drone
    tello = Tello()
    tello.connect()

    # Initialize Hopfield Network with expected patterns.
    expected_patterns, weights = init_hopfield()

    # Start camera stream and execute computer vision.
    # cap = cv.VideoCapture(0)
    tello.streamon()
    while True:
        # _, frame = cap.read()
        frame = tello.get_frame_read().frame

        # TODO: Verify size of frame
        # frame = cv.resize(frame, (1280, 720))

        pattern = process_frame(frame)
        if pattern is not None and len(pattern) == hopfield_n:
            # print("Recognizing pattern...")
            recognize_pattern(pattern, expected_patterns, weights, tello)

        # TODO: IMPORTANT!!! Only recognize a pattern once, do not send 30 of the same command to the drone.

        cv.imshow("Stream", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            tello.streamoff()
            tello.land()
            break

    tello.end()
    # cap.release()
    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
