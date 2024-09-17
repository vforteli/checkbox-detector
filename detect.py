import cv2
from utils import check_angles_90_degrees


image = cv2.imread('test_images/form.png', cv2.IMREAD_GRAYSCALE)
# image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)
blurred = cv2.GaussianBlur(image, (5, 5), 0)  # get rid of noise

# Apply thresholding to get a binary image
_, binary_image = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("processed image", binary_image)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(
    binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

for i, contour in enumerate(contours):
    epsilon = 0.03 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # if we have 4 points and the and the angles are all roughly 90 degrees, this is probably a rectangle \o/
    if len(approx) == 4 and check_angles_90_degrees(approx, tolerance=20):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.polylines(output_image, approx, True, (0, 255, 0))

        # Filter based on the aspect ratio and size (to match checkbox dimensions)
        aspect_ratio = w / float(h)
        if 0.8 <= aspect_ratio <= 1.2 and 20 <= w <= 80 and 20 <= h <= 80:
            roi_inside_box = binary_image[y:y+h, x:x+w]

            # Count non-zero pixels inside the ROI (to detect if it's filled)
            non_zero_pixels = cv2.countNonZero(roi_inside_box)
            total_pixels = w * h
            fill_ratio = non_zero_pixels / total_pixels

            color = (0, 0, 255) if fill_ratio > 0.8 else (0, 255, 0)

            cv2.rectangle(output_image, (x, y),
                          (x + w, y + h), color, 2)


cv2.imshow("maybe checkboxes?", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
