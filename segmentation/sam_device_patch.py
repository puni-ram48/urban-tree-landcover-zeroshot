"""
SAM device compatibility patch: fixes batched_nms device mismatch error.
Import before segment_anything: import sam_device_patch
"""

import torch
import torchvision
import torchvision.ops
import torchvision.ops.boxes as box_ops

original_batched_nms = box_ops.batched_nms

def patched_batched_nms(boxes, scores, idxs, iou_threshold):
    device = boxes.device
    boxes = boxes.to(device)
    scores = scores.to(device)
    idxs = idxs.to(device)
    return original_batched_nms(boxes, scores, idxs, iou_threshold)

box_ops.batched_nms = patched_batched_nms
torchvision.ops.batched_nms = patched_batched_nms
