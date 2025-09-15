import numpy as np      #not used?
import cv2


def print_image_information(image):
    #shape -> (height, width, channels)
    #The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color):
    height, width, channels = image.shape

    print("Height:", height)
    print("Width:", width)
    print("Channels:", channels)
        #The shape of an image is accessed by img.shape. It returns a tuple of the number of rows, columns, and channels (if the image is color):
    print("Size:", image.size)
        #could also be made with height * width
    print("Data type:", image.dtype)
        #img.dtype is very important while debugging because a large number of errors in OpenCV-Python code are caused by invalid datatype.


def main():
    print("Hello, World!")

    #Load image (1 = color)
    image = cv2.imread("../lena-1.png", 1)

    if image is None:
        print("Error: Image not found!")
        return

    #Print information
    print_image_information(image)

    #Show image
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
