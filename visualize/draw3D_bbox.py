import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
import yaml
import pandas as pd
from kitti_util import *
from matplotlib.lines import Line2D
import cv2
from tqdm import tqdm
from einops import rearrange


def compute_3d_box_cam2(h, w, l, x, y, z, yaw):
    """
    Return : 3xn in cam2 coordinate
    """
    R = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0,          1,          0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d_cam2 += np.vstack([x, y, z])
    return corners_3d_cam2


def read_detection(path, gt=False):
    try:
        df = pd.read_csv(path, header=None, sep=' ')
    except pd.errors.EmptyDataError:
        return []
    if gt:
        df[len(df.columns)] = 0.8

    df.columns = [
        'type', 'truncated', 'occluded', 'alpha',
        'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
        'height', 'width', 'length',
        'pos_x', 'pos_y', 'pos_z',
        'rot_y', 'score'
    ]

    df = df[df['type'] == 'Car']
    df.reset_index(drop=True, inplace=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Project GT & predicted 3D boxes to image and save visualization."
    )

    parser.add_argument(
        "--valid_ids_path",
        type=str,
        default=None,
        help="The txt directory containing the desired image numbers."
    )

    parser.add_argument(
        "--print_info",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="./viz"
    )

    parser.add_argument(
        "--max_num",
        type=int,
        default=-1,
        help="Number of images to use."
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    return args


def main(args):

    os.makedirs(args.out_dir, exist_ok=True)


    if args.dir is None:
        cfg = yaml.load(open('../configs/monoclue.yaml', 'r'), Loader=yaml.Loader)
        args.dir = cfg['dataset']['root_dir']
        args.dir = os.path.join(args.dir, 'training')

    if args.valid_ids_path is None:
        args.valid_ids_path = '../outputs/monoclue/outputs/data'
        valid_ids = [int(line.split('.')[0]) for line in os.listdir(args.valid_ids_path)]
    else:
        with open(args.valid_ids_path, 'r') as f:
            valid_ids = [int(line.strip()) for line in f if line.strip().isdigit()]

    if args.max_num > 0:
        valid_ids = valid_ids[:args.max_num]

    for img_id in tqdm(valid_ids):
        calib_path = os.path.join(args.dir,'calib', f"{img_id:06d}.txt")
        img_path   = os.path.join(args.dir,'image_2', f"{img_id:06d}.png")
        pred_path  = os.path.join('../outputs/monoclue/outputs/data',  f"{img_id:06d}.txt")
        gt_path    = os.path.join(args.dir, 'label_2' , f"{img_id:06d}.txt")

        calib = Calibration(calib_path)
        df_pred = read_detection(pred_path)
        df_gt   = read_detection(gt_path, gt=True)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[WARN] Cannot read image {img_path}")
            continue

        if len(df_gt) == 0:
            continue

        for i in range(len(df_gt)):
            corners_3d_cam2 = compute_3d_box_cam2(
                *df_gt.loc[i, ['height', 'width', 'length',
                               'pos_x', 'pos_y', 'pos_z', 'rot_y']]
            )
            pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
            image = draw_projected_box3d(
                image, pts_2d,
                color=(144, 238, 144),
                thickness=2
            )
            save_img = image.copy()

        for q in range(len(df_pred)):
            corners_3d_cam2 = compute_3d_box_cam2(
                *df_pred.loc[q, ['height', 'width', 'length',
                                  'pos_x', 'pos_y', 'pos_z', 'rot_y']]
            )
            pts_2d = calib.project_rect_to_image(corners_3d_cam2.T)
            save_img = draw_projected_box3d(
                save_img, pts_2d,
                color=(0, 0, 255),
                thickness=2
            )

        if args.print_info:
            cv2.putText(
                save_img,
                f"MonoCLUE / GT : [ {len(df_pred)} / {len(df_gt)} ]",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 144, 30),  # White (255,255,255)
                2,
                cv2.LINE_AA
            )

        out_path = os.path.join(args.out_dir, f"{img_id:06d}.png")
        cv2.imwrite(out_path, save_img)


if __name__ == "__main__":
    args = parse_args()
    main(args)
