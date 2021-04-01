""" Module for calculate iou of two bounding boxes"""


def area(a):
    """ Calculates area of bounding box."""
    return (int(a[2]) - int(a[0])) * (int(a[3]) - int(a[1]))


def union(au, bu, area_intersection):
    """ ."""
    area_a = area(au)
    area_b = area(bu)
    return area_a + area_b - area_intersection


def intersection(ai, bi):
    """ Calculates area of intersection."""

    x = max(int(ai[0]), int(bi[0]))  # 110
    y = max(int(ai[1]), int(bi[1]))  # 244
    w = min(int(ai[2]), int(bi[2])) - x  # 1365 - 110
    h = min(int(ai[3]), int(bi[3])) - y  # 856 - 244
    if w < 0 or h < 0:
        return 0
    return w * h


def compute_iou(a, b):
    """
    Computes the Intersection over Union (IoU) of two bounding boxes.

    Example
    ----------
    compute_iou([xmin, ymin, xmax, ymax], {xmin, ymin, xmax, ymax})

    Parameters
    ----------
    a: [list, tuple]
        [xmin, ymin, xmax, ymax]
    b: [list, tuple]
        [xmin, ymin, xmax, ymax]

    Returns
    -------
    iou: float
        in [0, 1]
    area_i: int
    """

    a = [int(el) for el in a]
    b = [int(el) for el in b]

    if a[0] >= a[2] or a[1] >= a[3] or b[0] >= b[2] or b[1] >= b[3]:
        return 0.0

    # Area of intersection
    area_i = intersection(a, b)
    area_u = union(a, b, area_i)
    iou = float(area_i) / float(area_u + 1e-6)

    return iou, area_i


def compute_iou2(a, b):
    """
    Computes overlapped region between ground truth and detected bounding boxes.

    Parameters
    ----------
    a: list
        [xmin, ymin, xmax, ymax]
    b: list
        [xmin, ymin, xmax, ymax]

    Returns
    -------
    iou: float
        in [0, 1]
    area_i: int
    """

    # Area of intersection
    area_i = intersection(a, b)

    # Area of two bounding boxes
    area_a = area(a)
    area_b = area(b)
    result_or = (2 * area_i)/(area_a + area_b)

    return result_or, area_i
