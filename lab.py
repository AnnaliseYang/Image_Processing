#!/usr/bin/env python3

"""
6.101 Lab 2:
Image Processing 2
"""

# NO ADDITIONAL IMPORTS!
# (except in the last part of the lab; see the lab writeup for details)
import math
from PIL import Image


####### FUNCTIONS FROM LAST WEEK ########


def get_index(image, row, col):
    """
    given an image (dict) and a 2D pixel location specified by row and col,
    returns the index of the pixel in a flat list
    """
    return int((row * image["width"]) + col)


def get_pixel(image, row, col):
    """
    given an image (dict) and a 2D pixel location specified by row and col,
    returns the pixel value (color)
    """
    index = get_index(image, row, col)
    return image["pixels"][index]


def get_boundary_pixel(image, row, col, boundary_behavior):
    """
    args:
        - image: dictionary
        - row: row index (0 to image['height']-1)
        - col: column index (0 to image['width']-1)
        - boundary_behavior: "zero", "wrap", or "extend"

    returns the value of the pixel (int) if the pixel is in range,
    otherwise return the value of a valid alternate pixel specified by boundary_behavior
    """
    height = image["height"]
    width = image["width"]

    if row in range(height) and col in range(width):
        # for in-range pixels, return the value
        pixel = get_pixel(image, row, col)
        return pixel
    elif boundary_behavior == "zero":
        # treat every out-of-bounds pixel as having a value of 0
        return 0
    elif boundary_behavior == "extend":
        # extend the input image beyond its boundaries
        r = 0 if row < 0 else height - 1 if row >= height else row
        c = 0 if col < 0 else width - 1 if col >= width else col
        pixel = get_pixel(image, r, c)
        return pixel
    elif boundary_behavior == "wrap":
        # wrap the input image at its edges
        pixel = get_pixel(image, row % image["height"], col % image["width"])
        return pixel
    else:
        return None


def set_pixel(image, row, col, color):
    """
    - changes the color of a given pixel in an image
    - modifies the pixel value and returns None
    """
    index = get_index(image, row, col)
    image["pixels"][index] = color


def apply_per_pixel(image, func):
    """
    applies a function to each pixel of an image
    returns a new image
    """
    result = {
        "height": image["height"],
        "width": image["width"],
        "pixels": [],
    }
    result["pixels"] = [0] * (result["width"] * result["height"])
    for col in range(image["width"]):
        for row in range(image["height"]):
            color = get_pixel(image, row, col)
            new_color = func(color)
            set_pixel(result, row, col, new_color)
    return result


def inverted(image):
    """
    returns a new image with the value of each pixel inverted (x => 255-x)
    """
    return apply_per_pixel(image, lambda color: 255 - color)


# GENERAL HELPER FUNCTIONS


def copy_img(image):
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": image["pixels"].copy(),
    }


def change_pixels(image, pixels):
    """
    Returns a new image of the same size as the original, with the pixels changed
    (DOES NOT MODIFY THE INPUT)
    """
    return {
        "height": image["height"],
        "width": image["width"],
        "pixels": pixels,
    }


def apply_kernel(image, row, col, kernel, kernel_size, boundary_behavior):
    """
    applies a given kernel to a specified pixel in an image at location (row, col)
    returns the new value at that pixel
    """
    adj_pixels = []  # make a list of pixels covered by the kernel
    shift = int(kernel_size / 2)  # the offset of the border pixels
    for r in range(row - shift, row + shift + 1):
        for c in range(col - shift, col + shift + 1):
            adj_pixels.append(get_boundary_pixel(image, r, c, boundary_behavior))
    assert len(adj_pixels) == len(kernel), "lengths don't match!"

    pixel = 0
    for i, value in enumerate(kernel):  # apply kernel to the pixel
        if value != 0:
            pixel += adj_pixels[i] * value

    return pixel


def correlate(image, kernel, boundary_behavior):
    """
    Compute the result of correlating the given image with the given kernel.
    `boundary_behavior` will one of the strings "zero", "extend", or "wrap",
    and this function will treat out-of-bounds pixels as having the value zero,
    the value of the nearest edge, or the value wrapped around the other edge
    of the image, respectively.

    if boundary_behavior is not one of "zero", "extend", or "wrap", return
    None.

    Otherwise, the output of this function should have the same form as a 6.101
    image (a dictionary with "height", "width", and "pixels" keys), but its
    pixel values do not necessarily need to be in the range [0,255], nor do
    they need to be integers (they should not be clipped or rounded at all).

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.

    Kernel: An n by n square matrix with integer values, where n is any odd number.

    My Kernel Representation: List
        - Stores the values of an n by n kernel (ints) in a list with length n*n.
        - Values are listed in row-major order.

    """
    # compute the kernel side length
    kernel_size = math.sqrt(len(kernel))

    new_pixels = []
    # loop through each pixel in the image and apply the kernel
    for row in range(image["height"]):
        for column in range(image["width"]):
            pixel = apply_kernel(
                image, row, column, kernel, kernel_size, boundary_behavior
            )
            new_pixels.append(pixel)

    return {"height": image["height"], "width": image["width"], "pixels": new_pixels}


def round_and_clip_image(image):
    """
    Given a dictionary, ensure that the values in the "pixels" list are all
    integers in the range [0, 255].

    All values should be converted to integers using Python's `round` function.

    Any locations with values higher than 255 in the input should have value
    255 in the output; and any locations with values lower than 0 in the input
    should have value 0 in the output.
    """
    pixels = image["pixels"].copy()

    for i, pixel in enumerate(pixels):
        if isinstance(pixel, float):  # make pixel values integers
            pixels[i] = round(pixel)
        if pixel > 255:  # clip pixels to 255
            pixels[i] = 255
        if pixel < 0:  # clip pixels to 0
            pixels[i] = 0
    return {"height": image["height"], "width": image["width"], "pixels": pixels}


# KERNEL GENERATOR FUNCTIONS


def get_box_blur_kernel(n):
    """
    returns an n*n box kernel with identical values that sum to 1
    """
    value = 1 / (n * n)
    return [value] * (n * n)


def get_sharp_kernel(n):
    """
    returns an n*n kernel that sharpens an image when applied
    """
    kernel = [-1 / (n * n)] * (n * n)
    kernel[int(len(kernel) / 2)] += 2

    return kernel


# FILTERS


def blurred(image, kernel_size):
    """
    Return a new image representing the result of applying a box blur (with the
    given kernel size) to the given input image.

    This process should not mutate the input image; rather, it should create a
    separate structure to represent the output.
    """
    # first, create a representation for the appropriate n-by-n kernel (you may
    # wish to define another helper function for this)

    # then compute the correlation of the input image with that kernel

    # and, finally, make sure that the output is a valid image (using the
    # helper function from above) before returning it.

    kernel = get_box_blur_kernel(kernel_size)  # generate blurring kernel
    blurred_image = round_and_clip_image(
        correlate(image, kernel, boundary_behavior="extend")
    )  # correlate the image with the blurring kernel
    return blurred_image


def blurred_with_edge_effects(image, kernel_size, boundary_behavior):
    """
    (THIS FUNCTION IS FOR CREATING THE ALTERNATIVE BLURRED CATS ONLY)
    a modified version of "blurred" that takes in an additional parameter,
    boundary_behavior, for correlation
    """
    kernel = get_box_blur_kernel(kernel_size)
    blurred_image = round_and_clip_image(correlate(image, kernel, boundary_behavior))
    return blurred_image


def sharpened(image, kernel_size):
    """
    given an image and kernel size,
    apply a sharpening kernel of the given size to the image
    returns a sharpened copy of the image
    """
    kernel = get_sharp_kernel(kernel_size)
    sharpened_image = round_and_clip_image(
        correlate(image, kernel, boundary_behavior="extend")
    )
    return sharpened_image


def edges(image):
    """
    detects the edges of an input image by
        - correlating an image with two kernels, K_1 and K_2
        - computing the square root of the sum of squares of corresponding pixels
    returns a rounded and clipped copy of the resulting image
    """
    K_1 = [-1, -2, -1, 0, 0, 0, 1, 2, 1]
    K_2 = [-1, 0, 1, -2, 0, 2, -1, 0, 1]

    new_pixels = []
    for row in range(image["height"]):
        for column in range(image["width"]):
            # apply K_1 and K_2 to each pixel and compute O_rc
            O_1 = apply_kernel(image, row, column, K_1, 3, "extend")
            O_2 = apply_kernel(image, row, column, K_2, 3, "extend")
            O_rc = round(math.sqrt(O_1**2 + O_2**2))
            new_pixels.append(O_rc)

    output_image = round_and_clip_image(
        {"height": image["height"], "width": image["width"], "pixels": new_pixels}
    )  # round and clip the image
    return output_image


####### FUNCTIONS FROM THIS WEEK ########

# VARIOUS FILTERS


def color_filter_from_greyscale_filter(filt):
    """
    Given a filter that takes a greyscale image as input and produces a
    greyscale image as output, returns a function that takes a color image as
    input and produces the filtered color image.
    """

    def isolate_color(image, color):
        """
        takes in an image, color (str) 'red', 'green', 'blue'
        returns a new image dictionary with pixel values from one color ("red")
        output image is in grayscale
        """
        assert color in ["red", "green", "blue"]
        color_index = 0 if color == "red" else 1 if color == "green" else 2
        pixels = [pix[color_index] for pix in image["pixels"]]
        return {"height": image["height"], "width": image["width"], "pixels": pixels}

    def color_filter(image):
        """
        function that takes in a color image and returns a filtered image
        """
        # separates pixels of each color and filter them as black and white images
        red = filt(isolate_color(image, "red"))
        green = filt(isolate_color(image, "green"))
        blue = filt(isolate_color(image, "blue"))

        # combine the three separate images after filtering them
        rgb_pixels = []
        for i in range(len(image["pixels"])):
            rgb_pixels.append((red["pixels"][i], green["pixels"][i], blue["pixels"][i]))

        out = {"height": image["height"], "width": image["width"], "pixels": rgb_pixels}
        return out

    return color_filter


def make_blur_filter(kernel_size):
    def blur_filter(image):
        return blurred(image, kernel_size)

    return blur_filter


def make_sharpen_filter(kernel_size):
    def sharpen_filter(image):
        return sharpened(image, kernel_size)

    return sharpen_filter


def filter_cascade(filters):
    """
    Given a list of filters (implemented as functions on images), returns a new
    single filter such that applying that filter to an image produces the same
    output as applying each of the individual ones in turn.
    """

    def chain_of_filters(image):
        img = copy_img(image)
        for filt in filters:
            img = filt(img)
        return img

    return chain_of_filters


# SEAM CARVING

# Main Seam Carving Implementation


def seam_carving(image, ncols):
    """
    Starting from the given image, use the seam carving technique to remove
    ncols (an integer) columns from the image. Returns a new image.
    """
    img = copy_img(image)

    def remove_min_energy_seam(image):
        # remove n cols of min-energy seams from the image
        energy = compute_energy(greyscale_image_from_color_image(image))
        cem = cumulative_energy_map(energy)
        seam = minimum_energy_seam(cem)
        return image_without_seam(image, seam)

    for _ in range(ncols):
        img = remove_min_energy_seam(img)
    return img


# Optional Helper Functions for Seam Carving


def greyscale_image_from_color_image(image):
    """
    Given a color image, computes and returns a corresponding greyscale image.

    Returns a greyscale image (represented as a dictionary).
    """
    grayscale_pixels = []  # create a new list of grayscale pixels
    for r, g, b in image["pixels"]:
        # compute grayscale value from rgb values
        grayscale_val = round(0.299 * r + 0.587 * g + 0.114 * b)
        grayscale_pixels.append(grayscale_val)

    return change_pixels(image, grayscale_pixels)


def compute_energy(grey):
    """
    Given a greyscale image, computes a measure of "energy", in our case using
    the edges function from last week.

    Returns a greyscale image (represented as a dictionary).
    """
    return edges(grey)


def get_adj_pixels_above(image, row, col):
    """
    returns a list of the three adjacent pixel values in the row above
    """
    adj_pixels = [
        get_boundary_pixel(image, row - 1, col - 1, "extend"),
        get_boundary_pixel(image, row - 1, col, "extend"),
        get_boundary_pixel(image, row - 1, col + 1, "extend"),
    ]
    return adj_pixels


def cumulative_energy_map(energy):
    """
    Given a measure of energy (e.g., the output of the compute_energy
    function), computes a "cumulative energy map" as described in the lab 2
    writeup.

    Returns a dictionary with 'height', 'width', and 'pixels' keys (but where
    the values in the 'pixels' array may not necessarily be in the range [0,
    255].
    """
    cem = copy_img(energy)
    for row in range(1, energy["height"]):
        for col in range(energy["width"]):
            # compute the cumulative energies of the 3 pixels above
            adj_cumulatives = get_adj_pixels_above(cem, row, col)
            cem["pixels"][get_index(cem, row, col)] += min(adj_cumulatives)
    return cem


def minimum_energy_seam(cem):
    """
    Given a cumulative energy map, returns a list of the indices into the
    'pixels' list that correspond to pixels contained in the minimum-energy
    seam (computed as described in the lab 2 writeup).
    """
    path = []  # stores the minimum cumulative energy path

    # locate the minimum pixel in the bottom row
    bottom_pixels = [
        get_pixel(cem, cem["height"] - 1, col) for col in range(cem["width"])
    ]
    r, c = (cem["height"] - 1, bottom_pixels.index(min(bottom_pixels)))
    path.append((r, c))

    def backtrack(row, col):
        """returns the row and column for the previous pixel in the path"""
        adj_cumulatives = get_adj_pixels_above(cem, row, col)

        # get coordinates for the pixel with the minimum adjacent cumulative energy
        prev_row = row - 1
        prev_col = col + (adj_cumulatives.index(min(adj_cumulatives)) - 1)

        # clip if out of range
        prev_col = 0 if prev_col < 0 else prev_col
        return prev_row, prev_col

    while len(path) < cem["height"]:
        r, c = path[-1]
        path.append(backtrack(r, c))

    path.reverse()
    for i, loc in enumerate(path):
        row, col = loc
        path[i] = get_index(cem, row, col)
    return path


def image_without_seam(image, seam):
    """
    Given a (color) image and a list of indices to be removed from the image,
    return a new image (without modifying the original) that contains all the
    pixels from the original image except those corresponding to the locations
    in the given list.
    """
    new_pixels = image["pixels"].copy()
    for i in sorted(seam, reverse=True):
        # remove the seam indices in reverse order
        del new_pixels[i]
    out = {"height": image["height"], "width": image["width"] - 1, "pixels": new_pixels}
    return out


# CUSTOM FUNCTIONS


def get_all_adj_pixels(image, row, col):
    """
    returns the nine adjacent pixels around a given location
    """
    pixels = []
    for i in range(3):
        pixels += get_adj_pixels_above(image, row + i, col)
    return pixels


def mean(values):
    """
    returns the average of a list of values
    """
    return sum(values) / len(values)


def median(values):
    """
    returns the median of a list of values
    """
    sorted_list = sorted(values)
    return sorted_list[int(len(sorted_list) / 2)]


def get_neighbors_loc(image, row, col):
    neighbors = [(row + 1, col), (row - 1, col), (row, col + 1), (row, col - 1)]
    return [
        (row, col)
        for row, col in neighbors
        if row in range(image["height"]) and col in range(image["width"])
    ]

import time

def custom_feature(image, background=None, p=0.4, pos=(0, 0)):
    """
    removes the background of the input image,
    - takes in an optional 'background' image to replace the original background
        - if no background provided, replace with white pixels
    - p is a cutoff parameter (float) ranging from 0 to 1.
        - Lower energy parts of the image are detected as the background.
        - defaults to 0.4
    - pos is an optional argument letting the user specify
    the location of a starting pixel (row, col) anywhere in the background
        - defaults to the top left corner of the image
    returns a new image with the background changed
    """
    start = time.time()

    def remove_background():
        """
        removes the background of the input image
        returns a new image with the background removed and replaced by white pixels
        """
        img = greyscale_image_from_color_image(image)
        energy = compute_energy(img)
        max_energy = max(
            energy["pixels"]
        )  # the maximum energy in the picture (identifies edges)
        cutoff = (
            p * max_energy
        )  # scale the maximum energy by p to compute a cutoff value.

        to_check = [pos]  # initialize the to_check queue with the starting pixel
        visited = {pos}  # records all visited pixels
        # visited_list = []

        new_pixels = image["pixels"].copy()
        while to_check:
            current_pos = to_check.pop(
                0
            )  # set the current position as the first in queue
            if get_boundary_pixel(energy, *current_pos, "extend") < cutoff:
                new_pixels[get_index(image, *current_pos)] = (255, 255, 255)
                for neighbor in get_neighbors_loc(image, *current_pos):
                    if neighbor not in visited:
                        to_check.append(neighbor)
                        visited.add(neighbor)
                        # visited_list.append(neighbor)

        # image after one round of background removal
        no_background_img = change_pixels(image, new_pixels)

        def remove_trace_background(no_background_img):
            """
            takes in the output image from remove_background
            removes any remaining background pixels
            """
            image = no_background_img
            new_pixels = image["pixels"].copy()
            for row in range(image["height"]):
                for col in range(image["width"]):
                    color = new_pixels[get_index(image, row, col)]
                    if color != (255, 255, 255) and sum(color) < 700:
                        neighbors = get_neighbors_loc(image, row, col)
                        # remove an isolated pixel if most of its neighbors were removed
                        if median([get_pixel(image, *n) for n in neighbors]) == (
                            255,
                            255,
                            255,
                        ):
                            new_pixels[get_index(image, row, col)] = (255, 255, 255)
            return change_pixels(image, new_pixels)

        for _ in range(3):
            no_background_img = remove_trace_background(no_background_img)
        return no_background_img

    def change_background():
        """
        replaces the background of the original image with 'background'
        """
        no_background_img = remove_background()
        new_pixels = []
        for row in range(image["height"]):
            for col in range(image["width"]):
                if get_pixel(no_background_img, row, col) == (255, 255, 255):
                    r, g, b = get_boundary_pixel(background, row, col, "wrap")
                    scale = 0.7  # make the background darker
                else:
                    r, g, b = get_pixel(image, row, col)
                    scale = 1.5  # make the object brighter
                new_pixels.append((r * scale, g * scale, b * scale))
        return change_pixels(image, new_pixels)

    round_and_clip_color_image = color_filter_from_greyscale_filter(
        round_and_clip_image
    )
    if background:
        result = round_and_clip_color_image(change_background())
    else:
        result = round_and_clip_color_image(remove_background())
    print('Time:', time.time() - start)


    if background:
        return result
    else:
        return result


# HELPER FUNCTIONS FOR LOADING AND SAVING COLOR IMAGES


def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}

def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def load_greyscale_image(filename):
    """
    Loads an image from the given file and returns an instance of this class
    representing that image.  This also performs conversion to greyscale.

    Invoked as, for example:
       i = load_greyscale_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img_data = img.getdata()
        if img.mode.startswith("RGB"):
            pixels = [
                round(0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]) for p in img_data
            ]
        elif img.mode == "LA":
            pixels = [p[0] for p in img_data]
        elif img.mode == "L":
            pixels = list(img_data)
        else:
            raise ValueError(f"Unsupported image mode: {img.mode}")
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}


def save_greyscale_image(image, filename, mode="PNG"):
    """
    Saves the given image to disk or to a file-like object.  If filename is
    given as a string, the file type will be inferred from the given name.  If
    filename is given as a file-like object, the file type will be determined
    by the 'mode' parameter.
    """
    out = Image.new(mode="L", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


if __name__ == "__main__":
    # code in this block will only be run when you explicitly run your script,
    # and not when the tests are being run.  this is a good place for
    # generating images, etc.

    cat = load_color_image("test_images/cat.png")
    no_background_cat = custom_feature(cat)
    save_color_image(no_background_cat, "no_background_cat.png")

    space = load_color_image("test_images/space.png")
    cat_in_space = custom_feature(cat, space)
    save_color_image(cat_in_space, "cat_in_space.png")
