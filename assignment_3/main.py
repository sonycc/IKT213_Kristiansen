import cv2
import numpy as np

#1 Sobel Edge Detection
def sobel_edge_detection(image):

    # Test variable to test the different types
    test = False

    # Grayscale and blur the immage
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    if test:
        ksizes = [-1, 1, 5, 7]
        for k in ksizes:
            # Sobel X
            sobelx = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=0, ksize=k)
            sobelx_abs = cv2.convertScaleAbs(sobelx)
            cv2.imwrite(f"{k}_sobelx_ksize.png", sobelx_abs)

            # Sobel Y
            sobely = cv2.Sobel(blur, cv2.CV_64F, dx=0, dy=1, ksize=k)
            sobely_abs = cv2.convertScaleAbs(sobely)
            cv2.imwrite(f"{k}_sobely_ksize.png", sobely_abs)

            # Combined (magnitude)
            sobel_magnitude = cv2.magnitude(sobelx, sobely)
            sobel_magnitude_abs = cv2.convertScaleAbs(sobel_magnitude)
            cv2.imwrite(f"{k}_sobelxy_ksize.png", sobel_magnitude_abs)

            print(f"Saved {k}_sobelx_ksize.png, {k}_sobely_ksize.png, {k}_sobelxy_ksize.png")
    else:
        # Assignment default: dx=1, dy=1, ksize=1
        sobel = cv2.Sobel(blur, cv2.CV_64F, dx=1, dy=1, ksize=1)
        sobel_abs = cv2.convertScaleAbs(sobel)
        cv2.imwrite("solutions/sobel_default.png", sobel_abs)
        print("Saved sobel_default.png")

#2 Canny Edge Detection
def canny_edge_detection(image, threshold_1, threshold_2):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)

    edges = cv2.Canny(blur, threshold_1, threshold_2)
    cv2.imwrite("solutions/canny_edges.png", edges)

#3 Template match
def template_match(image, template, threshold=0.84):

    # Test variable to test the different types
    test = False

    # Grayscale images
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_gray.shape[::-1]

    if test:
        last_count = 0
        template_w, template_h = template_gray.shape[::-1]
        template_diag = np.sqrt(template_w ** 2 + template_h ** 2)  # diagonal of the template

        for t in np.arange(1.0, 0.0, -0.01):
            res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= t)
            points = list(zip(*loc[::-1]))  # switch x and y

            # Filter out duplicates based on distance
            filtered_points = []
            # min_dist = int(max(template_w, template_h) * (1 - t))  # relative to threshold
            min_dist = int(template_diag * 1.5)  # increase radius to 1.5x template diagonal
            for pt in points:
                if all(np.linalg.norm(np.array(pt) - np.array(fp)) > min_dist for fp in filtered_points):
                    filtered_points.append(pt)

            count = len(filtered_points)
            if count > last_count:
                print(f"Threshold: {t:.2f}, Detections: {(count+1)}")
                last_count = count
                # Draw rectangles
                for pt in filtered_points:  # switch x and y
                    cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

                # Save result
                cv2.imwrite(f"tests/{t:.2f}_template_matched.png", image)
        return

    # Template matching
    res = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    loc = np.where(res >= threshold)

    # Draw rectangles
    for pt in zip(*loc[::-1]):  # switch x and y
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0, 0, 255), 2)

    # Save result
    cv2.imwrite("solutions/template_matched.png", image)

#4 Rezising
def resize(image, scale_factor: int = 2, up_or_down: str = "up"):
    resized_image = np.copy(image)

    # Validate inputs
    if up_or_down.lower() not in ["up", "down"]:
        raise ValueError("up_or_down must be either 'up' or 'down'")

    # Perform resizing using image pyramids
    for i in range(scale_factor):
        if up_or_down.lower() == "up":
            resized_image = cv2.pyrUp(resized_image)
        else:
            resized_image = cv2.pyrDown(resized_image)

    # Save result
    cv2.imwrite("solutions/lena_resized_pyramid.png", resized_image)
    return resized_image


if __name__ == "__main__":
    # Load images and template
    image = cv2.imread("../lambo.png")
    image_shapes = cv2.imread("../shapes-1.png")
    template = cv2.imread("../shapes_template.jpg")

    # Run Sobel
    sobel_edge_detection(image)

    # Run Canny
    canny_edge_detection(image, 50, 50)

    # Run Template matching
    template_match(image_shapes, template, 0.9)

    # Run Resizer
    resize(image, 2, "up")
