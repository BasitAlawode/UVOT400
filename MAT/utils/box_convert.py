import numpy as np


def box_convert(boxes: np.ndarray, in_fmt: str, out_fmt: str) -> np.ndarray:
    """
    Converts boxes from given in_fmt to out_fmt.
    Supported in_fmt and out_fmt are:

    'xyxy': boxes are represented via corners, x1, y1 being top left and x2, y2 being bottom right.
    This is the format that torchvision utilities expect.

    'xywh' : boxes are represented via corner, width and height, x1, y2 being top left, w, h being width and height.

    'cxcywh' : boxes are represented via centre, width and height, cx, cy being center of box, w, h
    being width and height.

    Args:
        boxes (np.ndarray[N, 4]): boxes which will be converted.
        in_fmt (str): Input format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh'].
        out_fmt (str): Output format of given boxes. Supported formats are ['xyxy', 'xywh', 'cxcywh']

    Returns:
        np.ndarray[N, 4]: Boxes into converted format.
    """

    allowed_fmts = ("xyxy", "xywh", "cxcywh", "kalman")
    if in_fmt not in allowed_fmts or out_fmt not in allowed_fmts:
        raise ValueError("Unsupported Bounding Box Conversions for given in_fmt and out_fmt")

    if in_fmt == out_fmt:
        return boxes.copy()

    if in_fmt != 'xyxy' and out_fmt != 'xyxy':
        # convert to xyxy and change in_fmt xyxy
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
        elif in_fmt == "kalman":
            boxes = _box_kalman_to_xyxy(boxes)
        in_fmt = 'xyxy'

    if in_fmt == "xyxy":
        if out_fmt == "xywh":
            boxes = _box_xyxy_to_xywh(boxes)
        elif out_fmt == "cxcywh":
            boxes = _box_xyxy_to_cxcywh(boxes)
        elif out_fmt == "kalman":
            boxes = _box_xyxy_to_kalman(boxes)

    elif out_fmt == "xyxy":
        if in_fmt == "xywh":
            boxes = _box_xywh_to_xyxy(boxes)
        elif in_fmt == "cxcywh":
            boxes = _box_cxcywh_to_xyxy(boxes)
        elif in_fmt == "kalman":
            boxes = _box_kalman_to_xyxy(boxes)
    return boxes


def _box_kalman_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """

    Args:
        boxes (np.ndarray[N, 4 or 8]): boxes in (cx, cy, w/h, h) or (cx, cy, w/h, h, vcx, vcy, vw/h, vh)
        format which will be converted.

    Returns:
        boxes (np.ndarray(N, 4)): boxes in (x1, y1, x2, y2) format.
    """

    b = boxes.copy()
    b[:, 2] = b[:, 2] * b[:, 3]  # (cx, cy, w/h, h) -> (cx, cy, w, h)
    b[:, 0:2] = b[:, 0:2] - b[:, 2:4] * 0.5  # (cx, cy, w, h) -> (x1, y1, w, h)
    b[:, 2:4] = b[:, 2:4] + b[:, 0:2] - 1  # (x1, y1, w, h) -> (x1, y1, x2, y2)

    if len(b.shape) > 2:
        b = b[:, :4, 0]
    return b


def _box_cxcywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (cx, cy, w, h) format to (x1, y1, x2, y2) format.
    (cx, cy) refers to center of bounding box
    (w, h) are width and height of bounding box
    Args:
        boxes (np.ndarray[N, 4]): boxes in (cx, cy, w, h) format which will be converted.

    Returns:
        boxes (np.ndarray(N, 4)): boxes in (x1, y1, x2, y2) format.
    """
    # We need to change all 4 of them so some temporary variable is needed.
    b = boxes.copy()
    b[:, 0:2] = b[:, 0:2] - b[:, 2:4] * 0.5  # (cx, cy, w, h) -> (x1, y1, w, h)
    b[:, 2:4] = b[:, 2:4] + b[:, 0:2] - 1  # (x1, y1, w, h) -> (x1, y1, x2, y2)

    return b


def _box_xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (x, y, w, h) format to (x1, y1, x2, y2) format.
    (x, y) refers to top left of bouding box.
    (w, h) refers to width and height of box.
    Args:
        boxes (np.ndarray[N, 4]): boxes in (x, y, w, h) which will be converted.

    Returns:
        boxes (np.ndarray[N, 4]): boxes in (x1, y1, x2, y2) format.
    """
    b = boxes.copy()
    b[:, 2:4] = b[:, 2:4] + b[:, 0:2] - 1  # (x1, y1, w, h) -> (x1, y1, x2, y2)
    return b


def _box_xyxy_to_kalman(boxes: np.ndarray) -> np.ndarray:
    """

    Args:
        boxes (np.ndarray(N, 4)): boxes in (x1, y1, x2, y2) format.

    Returns:
        boxes (np.ndarray[N, 4 or 8]): boxes in (cx, cy, w/h, h) or (cx, cy, w/h, h, vcx, vcy, vw/h, vh)
        format which will be converted.
    """

    b = boxes.copy()
    b[:, 2:4] = b[:, 2:4] - b[:, 0:2] + 1  # (x1, y1, x2, y2) -> (x1, y1, w, h)
    b[:, 0:2] = b[:, 0:2] + b[:, 2:4] * 0.5  # (x1, y1, w, h) -> (cx, cy, w, h)
    b[:, 2] = b[:, 2] / b[:, 3]  # (cx, cy, w, h) -> (cx, cy, w/h, h)

    if len(b.shape) < 3:
        b = b[..., np.newaxis]
    return b


def _box_xyxy_to_cxcywh(boxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (cx, cy, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (np.ndarray[N, 4]): boxes in (x1, y1, x2, y2) format which will be converted.

    Returns:
        boxes (np.ndarray(N, 4)): boxes in (cx, cy, w, h) format.
    """
    b = boxes.copy()
    b[:, 2:4] = b[:, 2:4] - b[:, 0:2] + 1  # (x1, y1, x2, y2) -> (x1, y1, w, h)
    b[:, 0:2] = b[:, 0:2] + b[:, 2:4] * 0.5  # (x1, y1, w, h) -> (cx, cy, w, h)

    return b


def _box_xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Converts bounding boxes from (x1, y1, x2, y2) format to (x, y, w, h) format.
    (x1, y1) refer to top left of bounding box
    (x2, y2) refer to bottom right of bounding box
    Args:
        boxes (np.ndarray[N, 4]): boxes in (x1, y1, x2, y2) which will be converted.

    Returns:
        boxes (np.ndarray[N, 4]): boxes in (x, y, w, h) format.
    """
    b = boxes.copy()
    b[:, 2:4] = b[:, 2:4] - b[:, 0:2] + 1  # (x1, y1, x2, y2) -> (x1, y1, w, h)
    return b
