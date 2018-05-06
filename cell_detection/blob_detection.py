import cv2
from .image_processing import invert


def detect_blobs(image, base_image=None, detect_white=True):
    if base_image is None:
        base_image = image
    if detect_white:
        image = invert(image)
    is_cv3 = cv2.__version__.startswith("3.")
    params = cv2.SimpleBlobDetector_Params()
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.9
    params.filterByInertia = False
    params.minInertiaRatio = 0.2
    params.filterByArea = True
    params.maxArea = 40
    if is_cv3:
        detector = cv2.SimpleBlobDetector_create(params)
    else:
        detector = cv2.SimpleBlobDetector(params)
    keypoints = detector.detect(image)
    im_with_keypoints = cv2.drawKeypoints(base_image, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, im_with_keypoints
