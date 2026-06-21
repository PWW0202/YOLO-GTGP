# ------------------------------------------------------------------------------
# 🚀 Advanced Soft-NMS & IoU Toolkit for PyTorch / Ultralytics YOLO
# ------------------------------------------------------------------------------
# This module provides a robust, tensor-safe implementation of Soft-NMS along 
# with a comprehensive suite of advanced Bounding Box Regression metrics 
# (GIoU, DIoU, CIoU, EIoU, SIoU).
# ------------------------------------------------------------------------------

import math
import torch

__all__ = ['bbox_iou_for_nms', 'soft_nms']

def bbox_iou_for_nms(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, eps=1e-7):
    """
    Calculate Intersection over Union (IoU) and various penalized IoUs for NMS processing.

    Args:
        box1 (torch.Tensor): A single bounding box tensor with shape (1, 4).
        box2 (torch.Tensor): Multiple bounding boxes tensor with shape (N, 4).
        xywh (bool): If True, input format is (x_center, y_center, width, height). 
                     If False, input format is (x1, y1, x2, y2). Default is False.
        GIoU (bool): Calculate Generalized IoU.
        DIoU (bool): Calculate Distance IoU.
        CIoU (bool): Calculate Complete IoU.
        EIoU (bool): Calculate Efficient IoU.
        SIoU (bool): Calculate Scylla IoU.
        eps (float): A small epsilon value to prevent division by zero.

    Returns:
        (torch.Tensor): Calculated IoU array of shape (N,) applying the specified penalties.
    """
    # Transform coordinates to (x1, y1, x2, y2) format
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Calculate Intersection Area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * \
            (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)

    # Calculate Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # Standard IoU
    iou = inter / union

    # Calculate Advanced Penalities
    if CIoU or DIoU or GIoU or EIoU or SIoU:
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # Convex width
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)  # Convex height
        
        if CIoU or DIoU or EIoU or SIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # Convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # Center distance squared
            
            if CIoU:
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
            elif EIoU:
                rho_w2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2
                rho_h2 = ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                cw2 = cw ** 2 + eps
                ch2 = ch ** 2 + eps
                return iou - (rho2 / c2 + rho_w2 / cw2 + rho_h2 / ch2)
            elif SIoU:
                s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps
                s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma_val = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha_1 = torch.abs(s_cw) / sigma_val
                sin_alpha_2 = torch.abs(s_ch) / sigma_val
                threshold = pow(2, 0.5) / 2
                sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
                angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
                rho_x = (s_cw / cw) ** 2
                rho_y = (s_ch / ch) ** 2
                gamma = angle_cost - 2
                distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
                omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
                omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
                shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps
            
            return iou - rho2 / c2  # DIoU
        
        c_area = cw * ch + eps
        return iou - (c_area - union) / c_area  # GIoU
        
    return iou


def soft_nms(bboxes, scores, iou_thresh=0.45, sigma=0.5, score_threshold=0.25):
    """
    Robust, tensor-safe implementation of Soft Non-Maximum Suppression.
    
    This function avoids index misalignment issues common in naive PyTorch 
    implementations by executing global in-place score updates followed 
    by global re-sorting at each iteration.

    Args:
        bboxes (torch.Tensor): Bounding boxes tensor of shape (N, 4).
        scores (torch.Tensor): Confidence scores tensor of shape (N,).
        iou_thresh (float): The IoU threshold where the Gaussian penalty is triggered. 
                            Default is 0.45.
        sigma (float): The variance of the Gaussian penalty function. 
                       Default is 0.5.
        score_threshold (float): The minimum score required to keep a box. 
                                 Default is 0.25.

    Returns:
        (torch.LongTensor): The indices of the bounding boxes to keep.
    """
    if bboxes.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=bboxes.device)

    # Clone scores to prevent destructive in-place modifications on original data
    scores = scores.clone()
    keep = []
    
    # Initial descending sort
    order = torch.argsort(scores, descending=True)

    while order.numel() > 0:
        # 1. Pop the index with the maximum score
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 2. Extract remaining indices
        remaining_indices = order[1:]

        # 3. Calculate standard IoU (STRICTLY STANDARD IoU for Gaussian penalty stability)
        ious = bbox_iou_for_nms(
            bboxes[i].unsqueeze(0), 
            bboxes[remaining_indices], 
            GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False
        ).view(-1)

        # 4. Identify boxes exceeding the overlap threshold
        overlap_mask = ious > iou_thresh

        if overlap_mask.any():
            # Get relative and absolute indices of penalized boxes
            penalize_idx = overlap_mask.nonzero(as_tuple=False).squeeze(-1)
            real_penalize_idx = remaining_indices[penalize_idx]
            penalize_ious = ious[penalize_idx]
            
            # Apply Gaussian decay: e^(-IoU^2 / sigma)
            weight = torch.exp(-(penalize_ious ** 2) / sigma)
            scores[real_penalize_idx] *= weight

        # 5. Prune boxes that fall below the score threshold
        survivors_mask = scores[remaining_indices] > score_threshold
        survivors_indices = remaining_indices[survivors_mask]

        if survivors_indices.numel() == 0:
            break

        # 6. Re-sort remaining survivors to guarantee the highest score is picked next
        survivors_scores = scores[survivors_indices]
        new_sort_idx = torch.argsort(survivors_scores, descending=True)
        order = survivors_indices[new_sort_idx]

    return torch.tensor(keep, dtype=torch.int64, device=bboxes.device)
