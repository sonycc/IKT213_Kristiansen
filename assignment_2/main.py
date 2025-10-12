import cv2
import numpy as np

#1. Padding
def padding(image, border_width=100):
    padded = cv2.copyMakeBorder(image, border_width, border_width, border_width, border_width,
                                cv2.BORDER_REFLECT)
    cv2.imwrite("solutions/lena_padded.png", padded)
    return padded

#2. Cropping
def crop(image, x0, x1, y0, y1):
        # i feel like this could be simplified by only having the x0 and y1 values, it needs to make that square anyhow.
        # Alternatively we could also have a starting point witdh and height
        # Alternatilvely we could have a center point with a distance from that in X and Y radius.

    cropped = image[y0:y1, x0:x1]
    cv2.imwrite("solutions/lena_cropped.png", cropped)
    return cropped

#3. Resize
def resize(image, width=200, height=200):
    resized = cv2.resize(image, (width, height))
    cv2.imwrite("solutions/lena_resized.png", resized)
    return resized

# 4. Manual Copy
def copy(image):
    height, width, channels = image.shape
    emptyPictureArray = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            emptyPictureArray[y, x] = image[y, x]
    cv2.imwrite("solutions/lena_copy.png", emptyPictureArray)
    return emptyPictureArray

# 5. Grayscale
def grayscale(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("solutions/lena_grayscale.png", gray)
    return gray

# 6. HSV conversion
def hsv(image):
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imwrite("solutions/lena_hsv.png", hsv_img)
    return hsv_img

# 7. Hue shift
def hue_shifted(image, hue=50):
    shifted = np.copy(image)

    # The hue shift is incorrect, possibly because you use % 256, which wraps the value. -Dominika Iza Kowalska at Sun Oct 5, 2025 7:11pmat Sun Oct 5, 2025 7:11pm
    shifted = (shifted.astype(np.int16) + hue) #% 256
    shifted = shifted.astype(np.uint8)
    cv2.imwrite("solutions/lena_hue_shifted.png", shifted)
    return shifted

# 8. Smoothing
def smoothing(image):
    blurred = cv2.GaussianBlur(image, (15, 15), 0, borderType=cv2.BORDER_DEFAULT)
    cv2.imwrite("solutions/lena_smoothed.png", blurred)
    return blurred

# 9. Rotation
def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv2.rotate(image, cv2.ROTATE_180)
    else:
        raise ValueError("Only 90 and 180 degrees are supported!")
        # the CV2 library also supports 270 but the assignment did not want that so here we are.
    cv2.imwrite(f"lena_rotated_{rotation_angle}.png", rotated)
    return rotated


if __name__ == "__main__":
    # Load image
    image = cv2.imread("../lena-1.png")

    # Run functions
    padded = padding(image)
    cropped = crop(image, 80, image.shape[1] - 130, 80, image.shape[0] - 130)
    resized = resize(image)
    copied = copy(image)
    gray = grayscale(image)
    hsv_img = hsv(image)
    hue_img = hue_shifted(image, 50)
    smooth_img = smoothing(image)
    rotated90 = rotation(image, 90)
    rotated180 = rotation(image, 180)
    # rotated270 = rotation(image, 270)
    # ^for testing