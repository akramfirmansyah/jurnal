from datetime import datetime
from constant.directory import images_dir

# import cv2
# import libcamera
# from picamera2 import Picamera2


def capture_image():
    """
    Captures an image using the camera and saves it to public/images directory.

    Returns:
        str: The filepath of the captured image if successful, None if failed.
    """

    filepath = f"{images_dir}" + datetime.now().strftime("%Y-%m-%d %H-%M-%S") + ".jpeg"

    # cam = Picamera2()
    # camera_config = cam.create_still_configuration(
    #     main={"size": (1920, 1080), "format": "RGB888"}
    # )
    # camera_config["transform"] = libcamera.Transform(hflip=1, vflip=1)
    # cam.configure(camera_config)

    # cam.start()

    # img = cam.capture_array()

    # isDone = cv2.imwrite(filepath, img)

    # cam.stop()

    # if isDone:
    #     return filepath
    # else:
    #     return None

    return filepath
