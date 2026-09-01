import cv2


IMAGE_PATH = "assets/test_images/component.png"


def load_image(image_path):
    """Load an image from disk."""
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    return image


def detect_edges(image):
    """Convert image to grayscale and detect edges."""

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    return edges


def find_component_contour(edges):
    """Find the most likely component contour."""

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError("No contours detected.")

    print(f"Contours detected: {len(contours)}")

    # Minimum area required for a component
    MIN_AREA = 1000

    # Remove very small contours
    filtered_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_AREA
    ]

    print(
        f"Contours after area filtering: "
        f"{len(filtered_contours)}"
    )

    if not filtered_contours:
        raise ValueError(
            "No contours large enough to be a component."
        )

    # Select the largest remaining contour
    largest_contour = max(
        filtered_contours,
        key=cv2.contourArea
    )

    return largest_contour, filtered_contours


def get_component_info(contour):
    """Calculate useful information about the component."""

    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(contour)

    points = {
        "top_left": (x, y),
        "top_right": (x + width, y),
        "bottom_left": (x, y + height),
        "bottom_right": (x + width, y + height)
    }

    return {
        "area": area,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "points": points
    }

def draw_component(image, contour, info):
    """Draw the detected component and its bounding box."""

    result = image.copy()

    # Component border
    cv2.drawContours(
        result,
        [contour],
        -1,
        (0, 255, 0),
        2
    )

    x = info["x"]
    y = info["y"]
    width = info["width"]
    height = info["height"]

    # Bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )

    # Four initial reference points
    points = [
        (x, y),
        (x + width, y),
        (x, y + height),
        (x + width, y + height)
    ]

    for point in points:

        cv2.circle(
            result,
            point,
            8,
            (0, 0, 255),
            -1
        )

    return result

def detect_component(image):
    """
    Detect the main component in an image.

    Returns:
        dict containing contour, bounding box information,
        and initial measurement points.
    """

    edges = detect_edges(image)

    largest_contour, filtered_contours = find_component_contour(
        edges
    )

    info = get_component_info(
        largest_contour
    )

    return {
        "contour": largest_contour,
        "info": info,
        "points": info["points"],
        "edges": edges
    }

def main():

    # 1. Load image
    image = load_image(IMAGE_PATH)

    # 2. Detect edges
    result = detect_component(image)

    info = result["info"]
    contour = result["contour"]

    print("\nComponent information:")
    print(f"Area: {info['area']:.2f} pixels²")
    print(f"X: {info['x']}")
    print(f"Y: {info['y']}")
    print(f"Width: {info['width']} pixels")
    print(f"Height: {info['height']} pixels")
    print("\nInitial measurement points:")

    for name, point in info["points"].items():
        print(f"{name}: {point}")

    # 5. Draw result
    display = draw_component(
        image,
        contour,
        info
    )

    # 6. Display
    cv2.imshow(
        "Original",
        image
    )

    cv2.imshow(
        "Edges",
        result["edges"]
    )

    cv2.imshow(
        "Component Detection",
        display
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()