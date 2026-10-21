import torch
import torch.nn as nn


class ChamferLoss(nn.Module):
    """Memory-bounded Chamfer distance for batched point clouds.

    Defaults match the original implementation:
    bidirectional Chamfer with Euclidean distance and mean reduction.
    """

    def __init__(
        self,
        reduction="mean",
        chunk_size=1024,
        max_pred_points=None,
        max_target_points=None,
        bidirectional=True,
        direction="pred_to_target",
        squared=False,
    ):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if direction not in {"pred_to_target", "target_to_pred"}:
            raise ValueError("direction must be 'pred_to_target' or 'target_to_pred'")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        self.reduction = reduction
        self.chunk_size = chunk_size
        self.max_pred_points = max_pred_points
        self.max_target_points = max_target_points
        self.bidirectional = bidirectional
        self.direction = direction
        self.squared = squared

    def forward(self, pred, target):
        """
        Args:
            pred:   Tensor[B, N, 3], predicted point cloud.
            target: Tensor[B, M, 3], target point cloud.
        """
        if pred.ndim != 3 or target.ndim != 3:
            raise ValueError("pred and target must have shape [B, N, C] and [B, M, C]")
        if pred.size(0) != target.size(0) or pred.size(-1) != target.size(-1):
            raise ValueError("pred and target must have the same batch size and coordinate dimension")

        pred = self._sample_points(pred, self.max_pred_points)
        target = self._sample_points(target, self.max_target_points)

        if self.bidirectional:
            pred_to_target = self._nearest_distance(pred, target).mean(dim=1)
            target_to_pred = self._nearest_distance(target, pred).mean(dim=1)
            loss = pred_to_target + target_to_pred
        elif self.direction == "target_to_pred":
            loss = self._nearest_distance(target, pred).mean(dim=1)
        else:
            loss = self._nearest_distance(pred, target).mean(dim=1)

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
            dist = torch.cdist(query_chunk, reference, p=2)
            min_dist = dist.min(dim=2).values
            if self.squared:
                min_dist = min_dist.pow(2)
            mins.append(min_dist)
        return torch.cat(mins, dim=1)


if __name__ == "__main__":
    batch_size = 2
    num_points_pred = 10000
    num_points_target = 10000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pred = torch.randn(batch_size, num_points_pred, 3, device=device)
    target = torch.randn(batch_size, num_points_target, 3, device=device)

    chamfer_loss = ChamferLoss(chunk_size=1024).to(device)
    loss = chamfer_loss(pred, target)
    print(f"Chamfer loss: {loss.item():.6f}")
