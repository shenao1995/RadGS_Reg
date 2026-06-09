import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    """Memory-bounded Chamfer distance for batched point clouds.

    The original implementation builds a full B x N x M distance tensor with
    torch.cdist. For RadGS training this is often the main bottleneck because
    N=10k and M can be 10k-100k.
    """

    def __init__(
        self,
        reduction="mean",
        chunk_size=2048,
        max_pred_points=None,
        max_target_points=4096,
        bidirectional=True,
        squared=True,
        eps=1e-12,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.max_pred_points = max_pred_points
        self.max_target_points = max_target_points
        self.bidirectional = bidirectional
        self.squared = squared
        self.eps = eps

    def forward(self, pred, target):
        """
        Args:
            pred:   Tensor[B, N, 3], predicted Gaussian centers.
            target: Tensor[B, M, 3], target point cloud.
        """
        if pred.ndim != 3 or target.ndim != 3 or pred.size(-1) != target.size(-1):
            raise ValueError("pred and target must have shape [B, N, C] and [B, M, C]")

        pred = self._sample_points(pred, self.max_pred_points)
        target = self._sample_points(target, self.max_target_points)

        pred_to_target = self._nearest_distance(pred, target).mean(dim=1)
        if self.bidirectional:
            target_to_pred = self._nearest_distance(target, pred).mean(dim=1)
            loss = pred_to_target + target_to_pred
        else:
            loss = pred_to_target

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    @staticmethod
    def _sample_points(points, max_points):
        if max_points is None or points.size(1) <= max_points:
            return points
        idx = torch.randperm(points.size(1), device=points.device)[:max_points]
        return points[:, idx]

    def _nearest_distance(self, query, reference):
        mins = []
        for start in range(0, query.size(1), self.chunk_size):
            query_chunk = query[:, start:start + self.chunk_size]
            dist2 = self._pairwise_squared_distance(query_chunk, reference)
            min_dist2 = dist2.min(dim=2).values
            if self.squared:
                mins.append(min_dist2)
            else:
                mins.append(min_dist2.clamp_min(self.eps).sqrt())
        return torch.cat(mins, dim=1)

    @staticmethod
    def _pairwise_squared_distance(x, y):
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1).unsqueeze(1)
        xy = torch.bmm(x, y.transpose(1, 2))
        return (x2 + y2 - 2.0 * xy).clamp_min_(0.0)


if __name__ == "__main__":
    batch_size = 2
    num_points_pred = 10000
    num_points_target = 20000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred = torch.randn(batch_size, num_points_pred, 3, device=device)
    target = torch.randn(batch_size, num_points_target, 3, device=device)

    chamfer_loss = ChamferLoss(chunk_size=1024, max_target_points=4096).to(device)
    loss = chamfer_loss(pred, target)
    print(f"Chamfer loss: {loss.item():.6f}")
