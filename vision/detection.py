import cv2
import numpy as np


MIN_AREA = 2000


def create_object_mask(image):
    """
    Create a binary mask representing the object.

    If the image has a useful alpha channel, use it directly.
    Otherwise, use grayscale thresholding and morphological cleanup.
    """

    # ---------------------------------------------------------
    # 1. Transparent PNG
    # ---------------------------------------------------------
    if image.shape[2] == 4:

        alpha = image[:, :, 3]

        # Check whether the alpha channel actually contains
        # transparency information.
        if np.min(alpha) < 250:

            _, mask = cv2.threshold(
                alpha,
                10,
                255,
                cv2.THRESH_BINARY
            )

            return clean_mask(mask)

    # ---------------------------------------------------------
    # 2. Normal RGB/BGR image
    # ---------------------------------------------------------
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    # Otsu automatically chooses a threshold.
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Try to make the object white.
    if np.mean(mask) > 127:
        mask = cv2.bitwise_not(mask)

    return clean_mask(mask)


def clean_mask(mask):
    """
    Remove small noise and close small gaps.
    """

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    return mask


def find_component_contour(mask):
    """
    Find the most likely component contour.
    """

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    candidates = []

    image_height, image_width = mask.shape

    image_area = image_width * image_height

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_AREA:
            continue

        # Ignore contours that are almost the entire image.
        if area > image_area * 0.90:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        # Ignore extremely small bounding boxes.
        if width < 20 or height < 20:
            continue

        # Calculate how solid the contour is.
        contour_area = cv2.contourArea(contour)
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        if hull_area == 0:
            continue

        solidity = contour_area / hull_area

        candidates.append(
            {
                "contour": contour,
                "area": area,
                "x": x,
                "y": y,
                "width": width,
                "height": height,
                "solidity": solidity,
            }
        )

    if not candidates:
        raise ValueError("No suitable component detected.")

    # Prefer large, reasonably solid objects.
    best = max(
        candidates,
        key=lambda item: (
            item["area"] * item["solidity"]
        )
    )

    return best["contour"], candidates


def get_component_info(contour):
    """
    Extract bounding box and measurement points.
    """

    area = cv2.contourArea(contour)

    x, y, width, height = cv2.boundingRect(contour)

    points = {
        "top_left": (x, y),
        "top_right": (x + width, y),
        "bottom_left": (x, y + height),
        "bottom_right": (x + width, y + height),
    }

    return {
        "area": area,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
        "points": points,
    }


def detect_component(image):
    """
    Main detection function used by the GUI.

    Returns:
        contour
        information
        measurement points
        object mask
    """

    if image is None:
        raise ValueError("Input image is empty.")

    mask = create_object_mask(image)

    contour, candidates = find_component_contour(mask)

    info = get_component_info(contour)

    return {
        "contour": contour,
        "info": info,
        "points": info["points"],
        "mask": mask,
        "candidates": candidates,
    }

def draw_component(image, contour, info):
    """
    Draw the detected component and its measurement points.
    """

    output = image[:, :, :3].copy()

    # Green contour
    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        4
    )

    x = info["x"]
    y = info["y"]
    width = info["width"]
    height = info["height"]

    # Blue bounding box
    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        4
    )

    # Red measurement points
    for point in info["points"].values():
        cv2.circle(
            output,
            point,
            8,
            (0, 0, 255),
            -1
        )

    return output


if __name__ == "__main__":

    image_path = "assets/test_images/component.png"

    image = cv2.imread(
        image_path,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load: {image_path}"
        )

    result = detect_component(image)

    info = result["info"]

    print("\nComponent detected")
    print("------------------")
    print(f"Area:   {info['area']:.2f} pixels²")
    print(f"X:      {info['x']}")
    print(f"Y:      {info['y']}")
    print(f"Width:  {info['width']} pixels")
    print(f"Height: {info['height']} pixels")

    print("\nPoints:")

    for name, point in info["points"].items():
        print(f"{name}: {point}")

    # Draw detection result.
    output = image[:, :, :3].copy()

    cv2.drawContours(
        output,
        [result["contour"]],
        -1,
        (0, 255, 0),
        3
    )

    x = info["x"]
    y = info["y"]
    width = info["width"]
    height = info["height"]

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (255, 0, 0),
        2
    )

    cv2.imshow(
        "VMS Detection Test",
        output
    )

    cv2.imshow(
        "Object Mask",
        result["mask"]
    )

    print("\nPress any key to close.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()