import math
import torch

# ==============================================================================
# YOLO-GTGP Post-Processing Modules
# Objective: Advanced bounding box suppression mechanisms tailored for GPR B-scans.
# Handles highly overlapping hyperbola signatures typical of adjacent geological 
# anomalies (e.g., clustered karst caves or fracture zones).
# ==============================================================================

def bbox_iou_for_nms(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False, ShapeIoU=False, eps=1e-7, scale=0.0):
    """
    Calculate Intersection over Union (IoU) and its advanced variants.
    Provides flexible distance, shape, and angle penalty metrics to accurately 
    evaluate bounding box overlap during the Non-Maximum Suppression phase.
    
    Args:
        box1 (torch.Tensor): Reference bounding box, shape (1, 4).
        box2 (torch.Tensor): Target bounding boxes, shape (n, 4).
        xywh (bool): If True, input format is (center_x, center_y, width, height).
        GIoU, DIoU, CIoU, EIoU, SIoU, ShapeIoU (bool): Flags to activate specific IoU penalties.
        eps (float): Small epsilon to prevent division by zero.
        scale (float): Scale parameter specifically used for ShapeIoU weighting.
        
    Returns:
        (torch.Tensor): Computed IoU variant scores.
    """
    # 1. Coordinate Transformation
    if xywh:
        # Convert (cx, cy, w, h) to (x1, y1, x2, y2)
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:
        # Input is already (x1, y1, x2, y2)
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # 2. Basic IoU Calculation
    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0) * (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    # Union area
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    # 3. Advanced IoU Penalty Calculations
    if CIoU or DIoU or GIoU or EIoU or SIoU or ShapeIoU:
        # Dimensions of the smallest enclosing convex box
        cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
        ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
        
        if CIoU or DIoU or EIoU or SIoU:
            c2 = cw ** 2 + ch ** 2 + eps  # Squared diagonal of the convex box
            # Squared distance between center points
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            
            if CIoU:
                # Complete IoU: Adds aspect ratio penalty
                v = (4 / math.pi ** 2) * (torch.atan(w2 / h2) - torch.atan(w1 / h1)).pow(2)
                with torch.no_grad(): 
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)
                
            elif EIoU:
                # Efficient IoU: Penalizes width and height differences separately
                rho_w2, rho_h2 = ((b2_x2 - b2_x1) - (b1_x2 - b1_x1)) ** 2, ((b2_y2 - b2_y1) - (b1_y2 - b1_y1)) ** 2
                return iou - (rho2 / c2 + rho_w2 / (cw ** 2 + eps) + rho_h2 / (ch ** 2 + eps))
                
            elif SIoU:
                # SCYLLA-IoU: Introduces angle cost to aid faster convergence
                s_cw, s_ch = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + eps, (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5 + eps
                sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
                sin_alpha = torch.where(torch.abs(s_cw) / sigma > pow(2, 0.5) / 2, torch.abs(s_ch) / sigma, torch.abs(s_cw) / sigma)
                gamma = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2) - 2
                distance_cost = 2 - torch.exp(gamma * (s_cw / cw) ** 2) - torch.exp(gamma * (s_ch / ch) ** 2)
                shape_cost = torch.pow(1 - torch.exp(-torch.abs(w1 - w2) / torch.max(w1, w2)), 4) + torch.pow(1 - torch.exp(-torch.abs(h1 - h2) / torch.max(h1, h2)), 4)
                return iou - 0.5 * (distance_cost + shape_cost) + eps
                
            elif ShapeIoU:
                # Shape-IoU: Focuses on shape and scale consistency
                ww = 2 * torch.pow(w2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                hh = 2 * torch.pow(h2, scale) / (torch.pow(w2, scale) + torch.pow(h2, scale))
                distance = (hh * (((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4) + ww * (((b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4)) / (cw ** 2 + ch ** 2 + eps)
                shape_cost = torch.pow(1 - torch.exp(-hh * torch.abs(w1 - w2) / torch.max(w1, w2)), 4) + torch.pow(1 - torch.exp(-ww * torch.abs(h1 - h2) / torch.max(h1, h2)), 4)
                return iou - distance - 0.5 * shape_cost
                
            return iou - rho2 / c2  # Default to Distance IoU (DIoU)
            
        # Generalized IoU (GIoU)
        return iou - (cw * ch + eps - union) / (cw * ch + eps)
        
    return iou  # Return standard IoU if no variants are selected


def soft_nms(bboxes, scores, iou_thresh=0.5, sigma=0.5, score_threshold=0.001):
    """
    Soft-NMS (Soft Non-Maximum Suppression) algorithm.
    
    Objective for GPR: In tunnel prediction, adjacent geological anomalies 
    often generate highly overlapping hyperbola signatures. Hard NMS would 
    falsely suppress these valid neighboring targets. Soft-NMS exponentially 
    decays the confidence score of overlapping boxes based on a Gaussian 
    function, improving the Recall rate for dense anomaly clusters.
    
    Args:
        bboxes (torch.Tensor): Predicted bounding boxes, shape (N, 4).
        scores (torch.Tensor): Corresponding confidence scores, shape (N,).
        iou_thresh (float): IoU threshold to trigger score decay.
        sigma (float): Variance of the Gaussian penalty function.
        score_threshold (float): Minimum score to retain a bounding box.
        
    Returns:
        (torch.LongTensor): Indices of the retained bounding boxes.
    """
    # Initialize a list of indices representing the current active boxes
    order = torch.arange(0, scores.size(0)).to(bboxes.device)
    keep = []
    
    while order.numel() > 0:
        # If only one box remains, add it and terminate
        if order.numel() == 1:
            keep.append(order[0].item())
            break
            
        # The box with the highest score is at index 0
        i = order[0]
        keep.append(i.item())
        
        # Calculate IoU between the max-score box and all remaining boxes
        # (Defaults to standard IoU here, but can be switched to SIoU/ShapeIoU if configured)
        iou = bbox_iou_for_nms(bboxes[i:i+1], bboxes[order[1:]]).squeeze()
        
        # Identify boxes that exceed the overlap threshold
        idx = (iou > iou_thresh).nonzero().squeeze()
        
        if idx.numel() > 0: 
            # Apply Gaussian decay penalty to highly overlapping boxes
            decay_penalty = torch.exp(-torch.pow(iou[idx], 2) / sigma)
            scores[order[idx + 1]] *= decay_penalty
        
        # Filter out boxes whose scores have decayed below the required threshold
        valid_mask = (scores[order[1:]] > score_threshold).nonzero().squeeze() 
        if valid_mask.numel() == 0: 
            break  # Stop if no boxes meet the threshold
            
        # Re-sort: Find the index of the new highest scoring box among the remaining
        max_score_idx = torch.argmax(scores[order[valid_mask + 1]]) 
        if max_score_idx != 0: 
            # Swap it to the front of the valid_mask array
            valid_mask[[0, max_score_idx]] = valid_mask[[max_score_idx, 0]]
            
        # Update the active order array for the next iteration
        order = order[valid_mask + 1]
    
    return torch.LongTensor(keep).to(bboxes.device)