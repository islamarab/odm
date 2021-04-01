""" Module for calculating metrics.
"""

from itertools import combinations

import pandas as pd

from .area_computation import compute_iou


def preprocess_df(df: pd.core.frame.DataFrame,
                  df_name: str = None,
                  save: bool = False) -> pd.core.frame.DataFrame:
    """ Deletes duplicates, add column id if there is not."""

    # delete duplicates
    df = df.drop_duplicates()

    # add column 'id' at left
    if 'id' not in df.columns:
        df.insert(0, "id", range(len(df)))

    if save:
        if df_name:
            df.to_csv(df_name, index=False)
        else:
            assert False, "df_name is None"

    return df


def remove_overlapping_objects(df: pd.core.frame.DataFrame) -> pd.core.frame.DataFrame:
    """ ."""

    unique_objects = []

    for image_name in df['image_id'].unique():
        dropped_objects = []
        objects = []

        sliced_df = df[df['image_id'] == image_name]
        objects_data = sliced_df.values.tolist()

        if len(objects_data) == 1:
            objects.append(objects_data[0])
        else:
            combinated_objects = combinations(objects_data, 2)
            for obj in combinated_objects:

                if any(el for el in obj if el in dropped_objects):
                    continue

                iou, _ = compute_iou(obj[0][2:6], obj[1][2:6])

                if iou:
                    if obj[0][-1] > obj[1][-1]:
                        dropped_objects.append(obj[1])
                        objects.append(obj[0])
                    else:
                        dropped_objects.append(obj[0])
                        objects.append(obj[1])
                else:
                    objects.extend(obj)

        unique_objects.extend([el for el in objects if el not in dropped_objects])

    result_df = pd.DataFrame(unique_objects, columns=df.columns)
    result_df = preprocess_df(result_df)

    return result_df


def calculate_metrics(actual_df: pd.core.frame.DataFrame,
                      detected_df: pd.core.frame.DataFrame,
                      prob_thresh: int = 0,
                      iou_thresh: float = 0.0):
    """ ."""

    objects = []

    # Annotated tables
    for i, actual_row in actual_df.iterrows():

        image_name = actual_row['image_id']
        detected_objects_df = detected_df[detected_df['image_id'] == image_name]

        old_iou = 0
        for j, det_row in detected_objects_df.iterrows():

            if det_row['xmin'] in ["Null", ""]:
                t_det_row = det_row
            else:
                bbox_actual = [actual_row['xmin'], actual_row['ymin'], actual_row['xmax'], actual_row['ymax']]
                bbox_det = [int(det_row['xmin']), int(det_row['ymin']), int(det_row['xmax']), int(det_row['ymax'])]
                iou, _ = compute_iou(bbox_actual, bbox_det)
                iou = round(iou, 2)

                if iou > old_iou and det_row['prob'] >= prob_thresh and iou > iou_thresh:
                    t_det_row = det_row
                    old_iou = iou

        if old_iou == 0:
            t_det_row = {"id": "Null", "xmin": "Null", "ymin": "Null", "xmax": "Null",
                         "ymax": "Null", "label": "Null", "prob": "Null"}

        objects.append((image_name,
                        actual_row["id"], actual_row["xmin"], actual_row["ymin"],
                        actual_row["xmax"], actual_row["ymax"], actual_row["label"],
                        t_det_row["id"], t_det_row["xmin"], t_det_row["ymin"], t_det_row["xmax"],
                        t_det_row["ymax"], t_det_row["label"], t_det_row["prob"],
                        old_iou))

    # Predicted tables
    detected_ids = [idx[7] for idx in objects]
    for i, det_row in detected_df.iterrows():
        if det_row["id"] not in detected_ids and det_row['prob'] >= prob_thresh and det_row["xmin"] != "Null":
            objects.append((det_row["image_id"], "Null", "Null", "Null", "Null", "Null", "Null",
                            det_row["id"], det_row["xmin"], det_row["ymin"],
                            det_row["xmax"], det_row["ymax"], det_row["label"], det_row["prob"], 0))

    # Df
    df = pd.DataFrame(objects, columns=["image_name",
                                        "a_id", "a_xmin", "a_ymin", "a_xmax", "a_ymax", "a_label",
                                        "d_id", "d_xmin", "d_ymin", "d_xmax", "d_ymax", "d_label", "d_prob",
                                        "iou"])

    return df


