import cv2 as cv
import numpy as np
import random


# Note: these parameters will need to change as the drone is farther from the camera, and the resolution of the video.
perimeter_thresh = 50
area_thresh = 2000


def detect_shape(cnt):
    perimeter = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
    return len(approx) == 4


def process_frame(frame):
    copy = frame.copy()

    gray = cv.cvtColor(copy, cv.COLOR_BGR2GRAY)
    cv.imshow("Gray", gray)

    gray = cv.GaussianBlur(gray, (7, 7), 0)

    thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 2)
    cv.imshow("Thresh", thresh)

    edges = cv.Canny(thresh, 100, 200)
    cv.imshow("Edges", edges)

    contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

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
    max_children_count = max(child_counts)
    max_children_cnt = contours[child_counts.index(max_children_count)]

    # Verify shape is a quadrilateral; this helps with bad detections (i.e., non-patterns), but still happens at times.
    if max_children_count > 0 and detect_shape(max_children_cnt):
        area = cv.contourArea(max_children_cnt)
        print(area)
        cv.drawContours(copy, [max_children_cnt], 0, (0, 255, 0), 4)

    cv.imshow("Output", copy)


def main():
    # TODO: Connect to Tello EDU Drone

    # TODO: Read Hopfield Network weights from CSV here.

    # TODO: Replace with the Tello EDU video stream
    cap = cv.VideoCapture(0)
    while cap.isOpened():
        _, frame = cap.read()

        # TODO: Verify size of frame
        # frame = cv.resize(frame, (1280, 720))

        process_frame(frame)

        cv.imshow("Stream", frame)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
