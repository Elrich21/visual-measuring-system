def calculate_scale(pixel_distance, real_distance):
    """
    Calculate pixels per real-world unit.

    Example:
        500 pixels / 100 mm = 5 pixels per mm
    """

    if real_distance <= 0:
        raise ValueError(
            "Real-world distance must be greater than zero."
        )

    return pixel_distance / real_distance


def pixels_to_real(pixel_distance, pixels_per_unit):
    """
    Convert a pixel measurement into a real-world measurement.
    """

    if pixels_per_unit <= 0:
        raise ValueError(
            "Scale must be greater than zero."
        )

    return pixel_distance / pixels_per_unit


if __name__ == "__main__":

    # Example calibration object
    known_pixel_distance = 500
    known_real_distance = 100  # mm

    scale = calculate_scale(
        known_pixel_distance,
        known_real_distance
    )

    print(f"Calibration scale: {scale:.2f} pixels/mm")

    # Test measurement
    measured_pixels = 531

    real_distance = pixels_to_real(
        measured_pixels,
        scale
    )

    print(
        f"Measurement: "
        f"{real_distance:.2f} mm"
    )