# Image Processor

This image processor includes various functions to process images, including filtering, blurring, sharpening, edge detection, and seam carving.

## Pillow Installation
Please install pillow using the following command:
```
pip install pillow
```

## Usage

To use the image processor, import the functions and apply them to images loaded using the helper functions.

## Example

```python
from lab import load_color_image, save_color_image, edges

image = load_color_image('path/to/image.png')
edges_image = edges(image)
save_color_image(edges_image, 'path/to/edges_image.png')
```

## Functions

### Pixel Access

  - `get_pixel(image, row, col)`
  - `set_pixel(image, row, col, color)`

### Image Manipulation

- `apply_per_pixel(image, func)`
- `inverted(image)`

### Correlation and Kernels

- `apply_kernel(image, row, col, kernel, kernel_size, boundary_behavior)`
- `correlate(image, kernel, boundary_behavior)`

### Blurring and Sharpening

- `blurred(image, kernel_size)`
- `sharpened(image, kernel_size)`

### Edge Detection

- `edges(image)`

### Custom Filter Functions

- `color_filter_from_greyscale_filter(filt)`
- `make_blur_filter(kernel_size)`
- `make_sharpen_filter(kernel_size)`
- `filter_cascade(filters)`

## Seam Carving

- `seam_carving(image, ncols)`

## Helper Functions

### Image Loading and Saving

- `load_color_image(filename)`
- `save_color_image(image, filename, mode="PNG")`
- `load_greyscale_image(filename)`

For detailed instructions, please refer to the individual function docstrings in lab.py.
