import cv2

def save_camera_info(filename="camera_outputs.txt"):

    # Open the default camera
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return


    # Get the default frame width and height
    frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))


    # Capture just one frame
    ret, frame = cam.read()
    if ret:
        out.write(frame)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)

    out.release()
    cam.release()
    cv2.destroyAllWindows()


def main():
    save_camera_info()

if __name__ == "__main__":
    main()
