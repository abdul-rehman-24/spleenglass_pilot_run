
import numpy as np
import torch

def dice_score_3d(pred_bin, gt_bin, eps=1e-6):
    intersection = (pred_bin * gt_bin).sum()
    return (2 * intersection + eps) / (pred_bin.sum() + gt_bin.sum() + eps)

@torch.no_grad()
def validate_3d(model, validation_ids, volume_cache, device):
    model.eval()
    dsc_list = []
    for vid in validation_ids:
        img_3d, lbl_3d = volume_cache.load(vid)
        S = img_3d.shape[-1]
        preds = np.zeros_like(lbl_3d)
        for s in range(S):
            x = torch.from_numpy(img_3d[:, :, s]).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x)
            prob = torch.sigmoid(logit)
            preds[:, :, s] = (prob.squeeze().cpu().numpy() > 0.5).astype(np.float32)
        dsc = dice_score_3d(preds, lbl_3d)
        dsc_list.append(dsc)
    model.train()
    return float(np.mean(dsc_list)), dsc_list
