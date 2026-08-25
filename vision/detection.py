import cv2

IMAGE_PATH = "assets/test_images/component.png"

# 1. Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

# 2. Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 3. Detect edges
edges = cv2.Canny(gray, 50, 150)

# 4. Find contours
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Contours detected: {len(contours)}")

# 5. Draw all detected contours
result = image.copy()

cv2.drawContours(
    result,
    contours,
    -1,
    (0, 255, 0),
    2
)

# 6. Display results
cv2.imshow("Original", image)
cv2.imshow("Edges", edges)
cv2.imshow("Contours", result)

cv2.waitKey(0)
cv2.destroyAllWindows()