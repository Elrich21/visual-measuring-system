import math


def calculate_distance(point1, point2):
    """
    Calculate the distance between two points in pixels.
    """

    return math.dist(point1, point2)


def calculate_width(points):
    """
    Calculate the horizontal distance between
    the selected left and right points.
    """

    return calculate_distance(
        points["top_left"],
        points["top_right"]
    )


def calculate_height(points):
    """
    Calculate the vertical distance between
    the selected top and bottom points.
    """

    return calculate_distance(
        points["top_left"],
        points["bottom_left"]
    )



if __name__ == "__main__":

    points = {
        "top_left": (29, 606),
        "top_right": (560, 606),
        "bottom_left": (29, 981),
        "bottom_right": (560, 981)
    }

    width = calculate_width(points)
    height = calculate_height(points)

    print(f"Width: {width:.2f} pixels")
    print(f"Height: {height:.2f} pixels")