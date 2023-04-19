import cv2 as cv
import numpy as np


# Note: these parameters will need to change as the drone is farther from the camera, and the resolution of the video.
perimeter_thresh = 50
area_thresh = 2000
cnt_approx_factor = 0.05


def detect_shape(cnt):
    perimeter = cv.arcLength(cnt, True)
    approx = cv.approxPolyDP(cnt, 0.04 * perimeter, True)
    return len(approx) == 4


def decode_pattern(frame, corners):
    pass


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

            # Decode the binary representation of the pattern.
            decode_pattern(frame, approx_cnt)

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
