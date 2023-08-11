from pathlib import Path


def y_from_filename(rotation_threshold, filename) -> str:
    """Extracts the direction label from the filename of an image.

    Example: "path/to/file/001_000011_-1p50.png" --> "right"
    """
    filename_stem = Path(filename).stem
    angle = float(filename_stem.split("_")[2].replace("p", "."))

    if angle > rotation_threshold:
        return "left"
    elif angle < -rotation_threshold:
        return "right"
    else:
        return "forward"
