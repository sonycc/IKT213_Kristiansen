import cv2

def save_camera_info(OutputFile):

    filename="solutions/"+OutputFile

    # Open the default camera
    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Cannot open camera")
        return




    # Get the default frame width and height
    video_fps = cam.get(cv2.CAP_PROP_FPS)
    frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
    frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)




    '''
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # TIL: fourcc stands for Four Character Code.
                                                    # It’s a 4-byte code that specifies the video codec (the compression format) OpenCV should use when writing the video.
    out = cv2.VideoWriter('solutions/output.mp4', fourcc, video_fps, (frame_width, frame_height))


    # Capture just one frame
    ret, frame = cam.read()
    if ret:
        out.write(frame)
        cv2.imshow('Camera', frame)
        cv2.waitKey(1)

    out.release()
    cam.release()
    cv2.destroyAllWindows()
    '''

    # Write the info into the file
    with open(filename, "w") as f:
        f.write(f"FPS: {video_fps}\n")
        f.write(f"Width: {frame_width}\n")
        f.write(f"Height: {frame_height}\n")

    # Release the camera
    cam.release()


def main():
    save_camera_info("camera_outputs.txt")

    # the programs throws the error:
    #[ WARN:0@0.186] global cap_gstreamer.cpp:1173 cv::GStreamerCapture::isPipelinePlaying OpenCV | GStreamer warning: GStreamer: pipeline have not been created
    # but it does what the tasks requires so i will ignore this for now against my better judgement.

if __name__ == "__main__":
    main()
