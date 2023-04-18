import cv2 as cv
import numpy as np


pattern_width = 10
pattern_height = 10


# For tuning color filtering
hsvBar = False


# Dummy function for getTrackbarPos
def nothing():
    return


def draw_grid(roi, w, h):
    unit_width = w / pattern_width
    unit_height = h / pattern_height

    grid = roi.copy()

    for i in range(pattern_width):
        for j in range(pattern_height):
            x = int(i * unit_width + unit_width / 2)
            y = int(j * unit_height + unit_height / 2)

            # print(grid[x][y])

            cv.circle(grid, (x, y), 3, (0, 0, 255), 1)
        print("New Unit\n")
    print("New Unit\n")

    cv.imshow("Grid", grid)


def process_frame(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    if hsvBar:
        l_h = cv.getTrackbarPos("LH", "Tracking")
        l_s = cv.getTrackbarPos("LS", "Tracking")
        l_v = cv.getTrackbarPos("LV", "Tracking")

        u_h = cv.getTrackbarPos("UH", "Tracking")
        u_s = cv.getTrackbarPos("US", "Tracking")
        u_v = cv.getTrackbarPos("UV", "Tracking")

    # Red HSV
    if not hsvBar:
        l_b = np.array([0, 172, 76])
        u_b = np.array([23, 255, 255])
    else:
        l_b = np.array([l_h, l_s, l_v])
        u_b = np.array([u_h, u_s, u_v])

    mask = cv.inRange(hsv, l_b, u_b)

    # cv2.imshow("mask", mask)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, cv.getStructuringElement(cv.MORPH_ELLIPSE, (10, 10)))
    # cv2.imshow("less noise", mask)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_ELLIPSE, (20, 20)))
    # cv2.imshow("filled gaps", mask)

    cv.imshow("testing", mask)

    # Object detection
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        max_idx = 0
        max_area = 0
        for i, cnt in enumerate(contours):
            if cv.contourArea(cnt) > max_area:
                max_idx = i
                max_area = cv.contourArea(cnt)

        print(max_area)
        if max_area > 100:

            # cnt = max(contours, key=lambda x: cv.contourArea(x))

            # Create a mask with contour.
            copy = frame.copy()
            cv.drawContours(copy, contours, max_idx, (0, 255, 0), 5)

            cv.imshow("test", copy)

            x, y, w, h = cv.boundingRect(contours[max_idx])
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

            ROI = frame[y:y+h, x:x+w]
            cv.imshow("roi", ROI)

            draw_grid(ROI, w, h)

    res = cv.bitwise_and(frame, frame, mask=mask)

    # cv.imshow("frame", frame)
    # cv.imshow("mask", mask)
    cv.imshow("res", res)


def main():
    # TODO: Connect to Tello EDU Drone

    # TODO: Read Hopfield Network weights from CSV here.

    if hsvBar:
        cv.namedWindow("Tracking")
        cv.createTrackbar("LH", "Tracking", 0, 255, nothing)
        cv.createTrackbar("LS", "Tracking", 0, 255, nothing)
        cv.createTrackbar("LV", "Tracking", 0, 255, nothing)
        cv.createTrackbar("UH", "Tracking", 255, 255, nothing)
        cv.createTrackbar("US", "Tracking", 255, 255, nothing)
        cv.createTrackbar("UV", "Tracking", 255, 255, nothing)

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
