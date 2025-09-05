import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
from segment_anything import sam_model_registry, SamPredictor
from lib.helpers.dataloader_helper import build_dataloader
import warnings
warnings.filterwarnings("ignore")
import yaml
import torch
from einops import rearrange
from tqdm import tqdm
import torch.nn.functional as F
import os

def prepare_targets(targets, batch_size):
    targets_list = []
    mask = targets['mask_2d']

    key_list = ['depth', 'boxes_2d']
    for bz in range(batch_size):
        target_dict = {}
        for key, val in targets.items():
            if key in key_list:
                target_dict[key] = val[bz][mask[bz]]

        targets_list.append(target_dict)
    return targets_list

sam_checkpoint = "./configs/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda:0"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)
predictor = SamPredictor(sam)

cfg = yaml.load(open('./configs/monoclue.yaml', 'r'), Loader=yaml.Loader)

if cfg['dataset']['type'] == 'waymo':
    img_path = os.path.join(cfg['dataset']['root_dir'], 'training', 'image')
    image_size = [768, 512]
else:
    img_path = os.path.join(cfg['dataset']['root_dir'], 'training', 'image_2')
    image_size = [1280, 384]

if len(cfg['dataset']['writelist']) > 1:
    out_path = os.path.join(cfg['dataset']['root_dir'],'training','label_sam_all')
else:
    out_path = os.path.join(cfg['dataset']['root_dir'],'training','label_sam')

obj_list = ['region','depth']
for obj in obj_list:
    os.makedirs(os.path.join(out_path, obj), exist_ok=True)

print(f'Image_path : {img_path}')
print(f'Save_path : {out_path}')

train_loader, test_loader = build_dataloader(cfg['dataset'])

visualize_png = False

for batch_idx, (_, calibs, targets, info) in enumerate(tqdm(train_loader)):
    # targets set
    calibs = calibs
    idx = info['img_id'][0].item()
    idx = str(idx).zfill(6)

    check_list = []
    if all(os.path.exists(os.path.join(out_path, obj, f"{idx}.npy")) for obj in obj_list):
        continue

    for key in targets.keys():
        targets[key] = targets[key]
    targets = prepare_targets(targets, 1)  # batch = 1

    # image set
    image = cv2.imread(img_path + f'/{idx}.png')
    w, h, _ = image.shape
    image_size_out = [h, w]
    image = cv2.resize(image, (image_size[0],image_size[1]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    predictor.set_image(image)

    bbox = targets[0]['boxes_2d']
    depth = targets[0]['depth']

    if len(bbox) > 0:
        offset = 3.
        bbox[:, 0] = torch.clamp(bbox[:, 0] - offset, min=0)  # xmin
        bbox[:, 1] = torch.clamp(bbox[:, 1] - offset, min=0)  # ymin
        bbox[:, 2] = torch.clamp(bbox[:, 2] + offset, max=image_size[0])  # xmax
        bbox[:, 3] = torch.clamp(bbox[:, 3] + offset, max=image_size[1])  # ymax

        transformed_boxes = predictor.transform.apply_boxes_torch(bbox.to(device=predictor.device), image_size)
        transformed_boxes.to(device=device)
        masks, _, _ = predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )

        masks= masks.to(torch.float32)
        masks = F.interpolate(masks, size=(image_size_out[1],image_size_out[0]), mode='nearest')
        masks = masks.squeeze(1).cpu().numpy().astype(np.float32)
        depth_mask = masks.copy()
        masks = np.where(masks == 0, np.inf, masks)
        masks = np.min(masks, axis=0)
        masks[np.isinf(masks)] = 0.
        masks = masks.astype(np.uint8)
        np.save(f'{out_path}/region/{idx}.npy', masks)
        if visualize_png:
            masks = masks.astype(np.uint8) * 255
            cv2.imwrite(f'{out_path}/region/{idx}.png', masks)

        depth = np.array(depth).astype(np.float32)
        depth = depth[:, :, None]
        depth_mask = depth_mask * depth
        depth_mask = np.where(depth_mask == 0, np.inf, depth_mask)
        depth_mask = np.min(depth_mask, axis=0)
        depth_mask[np.isinf(depth_mask)] = 0.
        np.save(f'{out_path}/depth/{idx}.npy', depth_mask)
        if visualize_png:
            depth_mask = depth_mask.astype(np.uint8)
            cv2.imwrite(f'{out_path}/depth/{idx}.png', depth_mask)
    else:
        masks = np.zeros((image_size_out[1], image_size_out[0])).astype(np.uint8)
        np.save(f'{out_path}/region/{idx}.npy', masks)
        depth_mask = np.zeros((image_size_out[1], image_size_out[0])).astype(np.float32)
        np.save(f'{out_path}/depth/{idx}.npy', depth_mask)