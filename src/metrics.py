
import numpy as np
import torch

def dice_score_3d(pred_bin, gt_bin, eps=1e-6):
    intersection = (pred_bin * gt_bin).sum()
    return (2 * intersection + eps) / (pred_bin.sum() + gt_bin.sum() + eps)

@torch.no_grad()
def validate_3d(model, validation_ids, volume_cache, device, batch_size=32):
    model.eval()
    dsc_list = []
    for vid in validation_ids:
        img_3d, lbl_3d = volume_cache.load(vid)
        S = img_3d.shape[-1]
        preds = np.zeros_like(lbl_3d)
        for start in range(0, S, batch_size):
            end = min(start + batch_size, S)
            batch = img_3d[:, :, start:end]
            batch = np.transpose(batch, (2, 0, 1))
            x = torch.from_numpy(batch).unsqueeze(1).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            preds[:, :, start:end] = (np.transpose(probs, (1, 2, 0)) > 0.5).astype(np.float32)
        dsc = dice_score_3d(preds, lbl_3d)
        dsc_list.append(dsc)
    model.train()
    return float(np.mean(dsc_list)), dsc_list
