import cv2
import numpy as np
from utils import check_angles_90_degrees


image = cv2.imread('test_images/jalmariorjan.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('test_images/lassipekka.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('test_images/form.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('test_images/image.png', cv2.IMREAD_GRAYSCALE)
output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
# blurred = cv2.GaussianBlur(image, (3, 3), 0)  # get rid of noise

kernel = np.ones((3, 3), np.uint8)
blurred = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)

cv2.imshow("blurred", image)
cv2.waitKey(0)

_, image = cv2.threshold(blurred, 120, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("binary", image)
cv2.waitKey(0)

# the test image is later used to figure out fill factor in detected checkboxes
test_image = image.copy()

# image = cv2.Canny(image, 50, 50, apertureSize=3)
# todo test hough lines...

# cv2.imshow("canny", image)
# cv2.waitKey(0)

# Define a kernel for extracting horizontal lines
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
horizontal_lines = cv2.erode(image, horizontal_kernel, iterations=1)
horizontal_lines = cv2.dilate(
    horizontal_lines, horizontal_kernel, iterations=1)

# Define a kernel for extracting vertical lines
vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
vertical_lines = cv2.erode(image, vertical_kernel, iterations=1)
vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=1)

image = cv2.add(horizontal_lines, vertical_lines)

image = cv2.morphologyEx(image, cv2.MORPH_CLOSE,
                         np.ones((2, 2), np.uint8), iterations=2)

# image = cv2.erode(image, np.ones((1, 1), np.uint8), iterations=1)
# image = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)

# cv2.imshow("lines?", vertical_lines)
# cv2.waitKey(0)

# cv2.imshow("lines?", horizontal_lines)
# cv2.waitKey(0)

cv2.imshow("lines?", image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(
    image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    cv2.polylines(output_image, [approx], True, (0, 140, 255))

    # if we have 4 points and the and the angles are all roughly 90 degrees, this is probably a rectangle \o/
    if len(approx) == 4 and check_angles_90_degrees(approx, tolerance=25):
        x, y, w, h = cv2.boundingRect(contour)
        # cv2.polylines(output_image, [approx], True, (0, 255, 0))

        # Filter based on the aspect ratio and size (to match checkbox dimensions)
        aspect_ratio = w / float(h)
        if 0.7 <= aspect_ratio <= 1.3 and 20 <= w <= 80 and 20 <= h <= 80:
            roi_inside_box = test_image[y:y+h, x:x+w]
            print(f"{y}:{y+h}, {x}:{x+w}")

            # Count non-zero pixels inside the ROI (to detect if it's filled)
            non_zero_pixels = cv2.countNonZero(roi_inside_box)
            total_pixels = w * h
            fill_ratio = non_zero_pixels / total_pixels

            color = (0, 0, 255) if fill_ratio > 0.2 else (0, 255, 0)

            cv2.rectangle(output_image, (x, y),
                          (x + w, y + h), color, 2)


cv2.imshow("maybe checkboxes?", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
