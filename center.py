import os
import glob
import numpy as np
import scipy.ndimage
from pathlib import Path
import imageio.v2 as imageio
from PIL import Image, ImageChops

from progress import Progress


INPUT_DIR = "faces"
OUTPUT_DIR = "straight_faces"


# exceptions to 2 vertical and 2 horizontal lines: (vertical, horizontal)
STANDARD_N_VERTICAL = 2
STANDARD_N_HORIZONTAL = 3
EXCEPTIONS = {
    "e-Z": (3, 3),
    "s-6": (3, 5),
    "s-U": (3, 5),
    "b": (3, 5)
}


# edge detection
SOBEL_X = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])
SOBEL_Y = SOBEL_X.T

# accumulation
THRESHOLD = 0.1
MIN_RATIO = 2

# line detection
# gauss filter radius
ACCU_FILTER_SIZE = 3
# radius of nearest neighbours getting eliminated
ACCU_NMP_RADIUS = 100

sigma = ACCU_FILTER_SIZE / 5
filter_offs = ACCU_FILTER_SIZE // 2
X = np.arange(-filter_offs, filter_offs + 1)
GAUSS_FILTER = np.exp(-X**2 / (2 * sigma * sigma))


def grayscale(img):
    return np.mean(img, axis=2)


def accumulate(edge_image, edges_x, edges_y):
    # vertical (horizontal) line: see where horizontal (vertical) change is above a threshold and where horizontal (vertical) change is much larger than the vertical (horizontal) change
    worthy_vertical = (edges_x > THRESHOLD) & (edges_y / np.maximum(edges_x, THRESHOLD) < 1 / MIN_RATIO)
    worthy_horizontal = (edges_y > THRESHOLD) & (edges_x / np.maximum(edges_y, THRESHOLD) < 1 / MIN_RATIO)
    # then accumulate the horizontal (vertical) change along the vertical (horizontal) axis where points could be a vertical (horizontal) line
    accu_vertical = np.sum(edge_image, where=worthy_vertical, axis=0)
    accu_horizontal = np.sum(edge_image, where=worthy_horizontal, axis=1)
    return accu_vertical, accu_horizontal


def detect_lines(acc, n):
    # Gaussian smoothing
    acc = scipy.ndimage.convolve(acc, GAUSS_FILTER, mode='constant', cval=0)
    # Detect lines with non-maximum suppression
    lines = []
    while (len(lines) < n) and np.any(acc):
        # Find point with highest scores
        i = np.argmax(acc)
        lines.append(i)
        # Eliminate non-maxima in the local neighborhood
        acc[max(0, i - ACCU_NMP_RADIUS):i + ACCU_NMP_RADIUS] = 0
    return lines


def center(original_image, n_vertical=2, n_horizontal=3):
    image = grayscale(original_image)
    image = np.asarray(image, dtype=np.float32) / 255

    edges_x = scipy.ndimage.convolve(image, SOBEL_X)
    edges_y = scipy.ndimage.convolve(image, SOBEL_Y)
    edge_image = np.sqrt(edges_x ** 2 + edges_y ** 2)

    # detect vertical and horizontal lines
    vertical_accumulator, horizontal_accumulator = accumulate(edge_image, edges_x, edges_y)
    vertical_lines = detect_lines(vertical_accumulator, n_vertical)
    horizontal_lines = detect_lines(horizontal_accumulator, n_horizontal)

    # find center of card (as middle of the surrounding rectangle)
    x_center = np.mean([np.min(vertical_lines), np.max(vertical_lines)])
    y_center = np.mean([np.min(horizontal_lines), np.max(horizontal_lines)])
    x_center_wanted = image.shape[1] // 2
    y_center_wanted = image.shape[0] // 2

    # shift card
    x_offset = round(x_center_wanted - x_center)
    y_offset = round(y_center_wanted - y_center)
    pil_image = Image.fromarray(original_image)
    pil_image = ImageChops.offset(pil_image, x_offset, y_offset)
    return np.array(pil_image)


def main():
    # create out dir
    if  not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    # straighten images
    paths = glob.glob(f"{INPUT_DIR}/*")
    progress = Progress(len(paths))
    for img_path in paths:
        # find out if card is an exception
        n_vertical, n_horizontal = STANDARD_N_VERTICAL, STANDARD_N_HORIZONTAL
        card = Path(img_path).stem
        if card in EXCEPTIONS.keys():
            n_vertical, n_horizontal = EXCEPTIONS[card]
        # center the card
        original_image = imageio.imread(img_path)
        straightened_image = center(original_image, n_vertical, n_horizontal)
        imageio.imsave(f"{OUTPUT_DIR}/{os.path.basename(img_path)}", straightened_image)
        progress.increment()


if __name__ == "__main__":
    main()
