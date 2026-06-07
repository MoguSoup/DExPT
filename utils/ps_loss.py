import torch
import torch.nn as nn


class PatchStructuralLoss(nn.Module):
    def __init__(self, model, patch_len_threshold=24, eps=1e-5):
        super().__init__()
        self.model = model
        self.patch_len_threshold = patch_len_threshold
        self.eps = eps
        self.kl_loss = nn.KLDivLoss(reduction='none')

    def forward(self, true, pred, step_mask=None):
        true_patch, pred_patch, patch_len, stride = self._adaptive_patching(true, pred)
        patch_mask = None
        if step_mask is not None:
            patch_mask = self._create_patches(step_mask.to(dtype=true.dtype), patch_len, stride).mean(
                dim=-1,
                keepdim=True,
            )
        corr_loss, var_loss, mean_loss = self._patch_wise_structural_loss(true_patch, pred_patch, patch_mask)
        alpha, beta, gamma = self._gradient_based_dynamic_weighting(true, pred, corr_loss, var_loss, mean_loss)
        return alpha * corr_loss + beta * var_loss + gamma * mean_loss

    def _create_patches(self, x, patch_len, stride):
        x = x.permute(0, 2, 1)
        batch_size, channels, length = x.shape
        patches = x.unfold(2, patch_len, stride)
        return patches.reshape(batch_size, channels, patches.shape[2], patch_len)

    def _adaptive_patching(self, true, pred):
        length = true.shape[1]
        true_fft = torch.fft.rfft(true, dim=1)
        frequency_list = torch.abs(true_fft).mean(0).mean(-1)
        if frequency_list.numel() > 1:
            frequency_list = frequency_list.clone()
            frequency_list[0] = 0.0
            top_index = int(torch.argmax(frequency_list).item())
        else:
            top_index = 1

        if top_index <= 0:
            period = length
        else:
            period = max(2, length // top_index)

        patch_len = min(max(2, period // 2), self.patch_len_threshold, length)
        stride = max(1, patch_len // 2)

        true_patch = self._create_patches(true, patch_len, stride)
        pred_patch = self._create_patches(pred, patch_len, stride)
        return true_patch, pred_patch, patch_len, stride

    def _patch_wise_structural_loss(self, true_patch, pred_patch, patch_mask=None):
        true_patch_mean = torch.mean(true_patch, dim=-1, keepdim=True)
        pred_patch_mean = torch.mean(pred_patch, dim=-1, keepdim=True)

        true_patch_var = torch.var(true_patch, dim=-1, keepdim=True, unbiased=False)
        pred_patch_var = torch.var(pred_patch, dim=-1, keepdim=True, unbiased=False)
        true_patch_std = torch.sqrt(true_patch_var + self.eps)
        pred_patch_std = torch.sqrt(pred_patch_var + self.eps)

        true_pred_patch_cov = torch.mean(
            (true_patch - true_patch_mean) * (pred_patch - pred_patch_mean),
            dim=-1,
            keepdim=True,
        )

        patch_linear_corr = (true_pred_patch_cov + self.eps) / (true_patch_std * pred_patch_std + self.eps)
        linear_corr_loss = self._masked_mean(1.0 - patch_linear_corr, patch_mask)

        true_patch_softmax = torch.softmax(true_patch, dim=-1)
        pred_patch_log_softmax = torch.log_softmax(pred_patch, dim=-1)
        var_loss = self._masked_mean(
            self.kl_loss(pred_patch_log_softmax, true_patch_softmax).sum(dim=-1, keepdim=True),
            patch_mask,
        )

        mean_loss = self._masked_mean(torch.abs(true_patch_mean - pred_patch_mean), patch_mask)
        return linear_corr_loss, var_loss, mean_loss

    def _masked_mean(self, value, mask):
        if mask is None:
            return value.mean()
        weight = mask.clamp(0.0, 1.0)
        denom = weight.sum().clamp_min(self.eps)
        return (value * weight).sum() / denom

    def _gradient_based_dynamic_weighting(self, true, pred, corr_loss, var_loss, mean_loss):
        true = true.permute(0, 2, 1)
        pred = pred.permute(0, 2, 1)
        true_mean = torch.mean(true, dim=-1, keepdim=True)
        pred_mean = torch.mean(pred, dim=-1, keepdim=True)
        true_var = torch.var(true, dim=-1, keepdim=True, unbiased=False)
        pred_var = torch.var(pred, dim=-1, keepdim=True, unbiased=False)
        true_std = torch.sqrt(true_var + self.eps)
        pred_std = torch.sqrt(pred_var + self.eps)
        true_pred_cov = torch.mean((true - true_mean) * (pred - pred_mean), dim=-1, keepdim=True)
        linear_sim = (true_pred_cov + self.eps) / (true_std * pred_std + self.eps)
        linear_sim = (1.0 + linear_sim) * 0.5
        var_sim = (2 * true_std * pred_std + self.eps) / (true_var + pred_var + self.eps)

        corr_norm = self._gradient_norm(corr_loss)
        var_norm = self._gradient_norm(var_loss)
        mean_norm = self._gradient_norm(mean_loss)
        gradient_avg_norm = (corr_norm + var_norm + mean_norm) / 3.0

        alpha = (gradient_avg_norm / (corr_norm + self.eps)).detach()
        beta = (gradient_avg_norm / (var_norm + self.eps)).detach()
        gamma = (gradient_avg_norm / (mean_norm + self.eps)).detach()
        gamma = gamma * torch.mean(linear_sim * var_sim).detach()
        return alpha, beta, gamma

    def _gradient_norm(self, loss):
        params = [p for p in self._head_parameters() if p.requires_grad]
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        norms = [grad.norm() for grad in grads if grad is not None]
        if not norms:
            return loss.new_tensor(1.0)
        return torch.stack(norms).mean()

    def _head_parameters(self):
        model = self.model.module if hasattr(self.model, 'module') else self.model
        if hasattr(model.net, 'structural_head_parameters'):
            return model.net.structural_head_parameters()
        return model.net.fc8.parameters()
