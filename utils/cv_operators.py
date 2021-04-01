""" Module for working with images"""

import os
import cv2 as cv


def cv_show(path_to_image, bboxes_dict, scale: float = 0.5, show_label: bool = False):
    """ Shows image with boundary boxes

    Parameters
    -----------
    path_to_image: str

    bboxes_dict: [list, dict]
        boundary boxes
    scale: float
        image scaling
    show_label: bool


    """

    window_name = "image"

    # Reading
    try:
        image = cv.imread(path_to_image)
    except Exception as e:
        print(e)

    if isinstance(bboxes_dict, dict):
        for color in bboxes_dict:
            bbox = bboxes_dict[color]
            for bb in bbox:
                if len(bb) > 0:
                    cv.rectangle(image,
                                 (int(bb[0]), int(bb[1])),
                                 (int(bb[2]), int(bb[3])),
                                 color=color,
                                 thickness=2)
                    if show_label:
                        cv.putText(image,
                                   bb[4],
                                   (int(bb[0]), int(bb[1])),
                                   cv.FONT_HERSHEY_COMPLEX_SMALL,
                                   1,
                                   (0, 0, 0),
                                   2)
    elif isinstance(bboxes_dict, list):
        for bb in bboxes_dict:
            if len(bb) > 0:
                cv.rectangle(image,
                             (int(bb[0]), int(bb[1])),
                             (int(bb[2]), int(bb[3])),
                             color=(0, 255, 0),
                             thickness=2)
                if show_label:
                    cv.putText(image,
                               bb[4],
                               (int(bb[0]), int(bb[1])),
                               cv.FONT_HERSHEY_COMPLEX_SMALL,
                               1,
                               (0, 0, 0),
                               2)

    # Resizing
    dsize = (int(image.shape[1]*scale), int(image.shape[0]*scale))
    output = cv.resize(image, dsize)
    cv.imshow(window_name, output)
    key = cv.waitKey(0)
    # if click on arrows on keyboard, then cut the file into a "problem images" folder
    # if key == 0:
    #     try:
    #         shutil.move(path_to_image, "problem images")
    #     except Exception as e:
    #         print(e)
    # # if click on any other button, then cut the file into a "right images" folder
    # else:
    #     try:
    #         shutil.move(path_to_image, "right images")
    #     except Exception as e:
    #         print(e)
    cv.destroyAllWindows()