import torch
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss(reduction="mean")

    def forward(self, student_masked_tokens, teacher_masked_tokens):
        """
        输入:
            student_masked_tokens: [B, M, D] 学生模型解码器输出的被mask位置的token
            teacher_masked_tokens: [B, M, D] 教师模型编码器输出的被mask位置的token

        输出:
            loss_rec: 标量
        """
        return self.criterion(student_masked_tokens, teacher_masked_tokens)


class ContrastiveAlignmentLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features_rgb, features_other):
        """
        输入:
            features_rgb: [N, D] RGB模态的patch特征
            features_other: [N, D] 另一模态的对应patch特征

        输出:
            loss_align: 标量
        """
        # L2归一化
        features_rgb = nn.functional.normalize(features_rgb, dim=-1)
        features_other = nn.functional.normalize(features_other, dim=-1)

        logits = (
            torch.matmul(features_rgb, features_other.T) / self.temperature
        )  # [N, N]
        labels = torch.arange(logits.shape[0]).to(logits.device)

        loss_i = nn.functional.cross_entropy(logits, labels)
        loss_t = nn.functional.cross_entropy(logits.T, labels)
        loss = (loss_i + loss_t) / 2

        return loss


class LinearHSICLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, X, Y):
        """
        输入:
            X: [N, D] 模态A的特征
            Y: [N, D] 模态B的特征

        输出:
            loss_hsic: 标量
        """
        n = X.size(0)

        # 中心化特征
        X = X - X.mean(dim=0)
        Y = Y - Y.mean(dim=0)

        # 计算协方差矩阵
        cov = torch.mm(X.t(), Y)

        # Frobenius范数平方
        loss = torch.norm(cov, p="fro") ** 2 / ((n - 1) ** 2)

        return loss


class AuxiliaryClassificationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, features, modality_labels):
        """
        输入:
            features: [N, D] 特征向量
            modality_labels: [N] 二值标签，0或1，表示来自哪个模态

        输出:
            loss_cls: 标量
        """
        logits = features.mean(dim=-1)  # 简单地取均值作为logit（可替换为MLP）
        targets = modality_labels.float()
        return self.criterion(logits, targets)
