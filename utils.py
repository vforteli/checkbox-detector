import math

import cv2


def vector_from_points(p1: cv2.typing.Point, p2: cv2.typing.Point) -> tuple[int, int]:
    return (p2[0] - p1[0], p2[1] - p1[1])


def dot_product(v1: cv2.typing.Point, v2: cv2.typing.Point) -> int:
    return v1[0] * v2[0] + v1[1] * v2[1]


def vector_magnitude(v: cv2.typing.Point) -> float:
    return math.sqrt(v[0]**2 + v[1]**2)


def angle_between(v1: cv2.typing.Point, v2: cv2.typing.Point) -> float:
    dot = dot_product(v1, v2)
    mag1 = vector_magnitude(v1)
    mag2 = vector_magnitude(v2)
    cos_theta = dot / (mag1 * mag2)
    # Clamp the value to avoid numerical issues
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle = math.acos(cos_theta)  # Angle in radians
    return math.degrees(angle)  # Convert to degrees


def matlike_to_points(points: cv2.typing.MatLike) -> list[cv2.typing.Point]:
    return [tuple(point[0]) for point in points]


def check_angles_90_degrees(points: cv2.typing.MatLike, angle: float = 90, tolerance: float = 10) -> bool:
    points = matlike_to_points(points)

    num_points = len(points)
    min_angle = angle - tolerance / 2
    max_angle = angle + tolerance / 2

    for i in range(num_points):
        p1 = points[i]
        p2 = points[(i + 1) % num_points]
        p3 = points[(i + 2) % num_points]

        v1 = vector_from_points(p1, p2)
        v2 = vector_from_points(p2, p3)

        angle = angle_between(v1, v2)

        if not (min_angle <= angle <= max_angle):
            return False

    return True
