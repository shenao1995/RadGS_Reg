import torch
import torch.nn as nn
import time


class ChamferLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChamferLoss, self).__init__()
        assert reduction in ['mean', 'sum', 'none'], "reduction must be 'mean', 'sum', or 'none'"
        self.reduction = reduction

    def forward(self, pred, target):
        """
        计算 Chamfer 损失，使用 torch.cdist 优化距离计算。

        参数:
            pred (torch.Tensor): 预测的点云，形状为 (B, N, 3)
            target (torch.Tensor): 目标点云，形状为 (B, M, 3)

        返回:
            torch.Tensor: Chamfer 损失。
        """
        # 计算所有预测点到目标点的距离
        # pred: (B, N, 3)
        # target: (B, M, 3)
        # torch.cdist 支持 batch 计算
        distances = torch.cdist(pred, target, p=2)  # (B, N, M)

        # 对每个预测点找到最近的目标点
        min_dist_pred, _ = torch.min(distances, dim=2)  # (B, N)

        # 对每个目标点找到最近的预测点
        min_dist_target, _ = torch.min(distances, dim=1)  # (B, M)

        # 计算 Chamfer 损失
        loss_pred = torch.mean(min_dist_pred, dim=1)  # (B,)
        loss_target = torch.mean(min_dist_target, dim=1)  # (B,)
        loss = loss_pred + loss_target  # (B,)

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss  # (B,)


if __name__ == '__main__':
    # 假设 batch_size=2, num_points_pred=20000, num_points_target=20000
    batch_size = 8
    num_points_pred = 20000
    num_points_target = 20000
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    pred = torch.randn(batch_size, num_points_pred, 3, device=device)
    target = torch.randn(batch_size, num_points_target, 3, device=device)

    chamfer_loss = ChamferLoss(reduction='mean', chunk_size=1000).to(device)
    loss = chamfer_loss(pred, target)
    print(f"Optimized Chamfer Loss: {loss.item()}")
