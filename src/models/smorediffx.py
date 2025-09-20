# =========================
# modules_x.py（可并入同文件或单独保存后导入）
# =========================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoiseRecipe(nn.Module):
    """
    结构化“造噪”：为边重建提供合成假阳性（内容相似但未交互）与度感知的真边脱落策略。
    仅采样索引用于loss，不改动图本身（由_ build_soft_R_once 完成软边）。
    """
    def __init__(self, n_users, n_items, topk_fake=10, drop_rate_hi=0.1, drop_rate_lo=0.02):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.topk_fake = topk_fake
        self.drop_rate_hi = drop_rate_hi
        self.drop_rate_lo = drop_rate_lo

    @torch.no_grad()
    def sample_fake_pos(self, users, R_csr, fusion_adj):
        """
        对每个 user，从 fusion 物品-物品图的近邻里采样 未交互 的“相似硬负例”作为假阳性。
        返回：dict[user] -> LongTensor[item_ids]
        """
        # R_csr: scipy CSR or torch sparse? 这里用 torch.sparse（coalesce）更简单
        f_adj = fusion_adj.coalesce()
        item_neighbors = {}
        idx = f_adj.indices()  # [2, nnz]
        val = f_adj.values()
        # 物品->近邻（按权排序）
        # 为效率，这里只做一次排序映射（可缓存）
        # 简化：使用topk on-the-fly（稀疏度通常够）
        res = []
        for u in users.tolist():
            # 用户已交互集合
            # 从 R_csr（原 self.R）提取u行的非零列
            # 这里假设 self.R 是 torch.sparse，外面传时用 coalesce()
            pass
        # 为避免冗长，此函数在调用方用更轻量的近邻抽样（见下 _edge_recon_pairs）
        return None

    @torch.no_grad()
    def _edge_recon_pairs(self, users, pos_items, neg_items, fusion_item_embeds, R_sparse, k_hard=5):
        """
        返回： (edge_pos_u, edge_pos_i, edge_neg_u, edge_neg_i)
        正样本：batch 内 (users, pos_items)
        假阳性（硬负例）：对每个 user，选和其正物品最相似的 k_hard 个物品（未交互且不等于正样本）
        """
        # 计算 batch 内 cond 向量
        pos_vec = F.normalize(fusion_item_embeds[pos_items], dim=-1)  # [B, d]
        all_items = F.normalize(fusion_item_embeds, dim=-1)           # [N, d]
        # 相似物品（硬负例）：对每个 pos，取 topk，排除 pos 本体；再合并去重
        with torch.no_grad():
            sim = torch.matmul(pos_vec, all_items.t())               # [B, N]
            sim.scatter_(1, pos_items.view(-1,1), -1.0)              # 去掉自身
            hard_vals, hard_idx = torch.topk(sim, k=k_hard, dim=1)   # [B, k]
        # 把 hard_idx 配到对应 users
        B = users.shape[0]
        edge_neg_u = users.view(-1,1).repeat(1, k_hard).reshape(-1)  # [B*k]
        edge_neg_i = hard_idx.reshape(-1)
        # 过滤已交互：与 R 的非零相交可选（成本高时可省）
        # 这里先返回原始对
        edge_pos_u = users
        edge_pos_i = pos_items
        return edge_pos_u, edge_pos_i, edge_neg_u, edge_neg_i


class DSPDenoiser(nn.Module):
    """
    三种可插拔降噪：FFT高SNR掩膜 / 1层Haar小波 / 图谱Chebyshev滤波
    mode: ['fft_mask', 'wavelet', 'cheby', 'none']
    """
    def __init__(self, mode='fft_mask', fft_keep_ratio=0.5, wavelet_thr=None,
                 cheby_theta=None, cheby_L=None):
        super().__init__()
        self.mode = mode
        self.fft_keep_ratio = fft_keep_ratio
        self.wavelet_thr = wavelet_thr
        # chebyshev
        self.cheby_theta = cheby_theta  # tensor([theta0..thetaK])
        self.cheby_L = cheby_L          # 归一化拉普拉斯（稀疏）

    def forward(self, z):  # z: [N, D]
        if self.mode == 'none':
            return z
        if self.mode == 'fft_mask':
            return self._fft_mask(z, self.fft_keep_ratio)
        if self.mode == 'wavelet':
            return self._wavelet_haar1(z, self.wavelet_thr)
        if self.mode == 'cheby':
            return self._cheby(z, self.cheby_L, self.cheby_theta)
        return z

    def _fft_mask(self, embeds, keep_ratio=0.5):
        fft = torch.fft.rfft(embeds, dim=1, norm='ortho')
        mag = torch.abs(fft)
        k = max(1, int(mag.size(1) * keep_ratio))
        th = torch.topk(mag, k, dim=1).values[:, -1:]          # 每行第k大阈
        mask = (mag >= th).to(fft.dtype)
        fft_m = fft * mask
        return torch.fft.irfft(fft_m, n=embeds.size(1), dim=1, norm='ortho')

    def _wavelet_haar1(self, x, thr=None):
        # 简化实现：Haar 1-level（D 偶数）
        N, D = x.shape
        if D % 2 == 1:
            x = F.pad(x, (0,1))
            D = D+1
        h = torch.tensor([2**-0.5, 2**-0.5], device=x.device, dtype=x.dtype).view(1,1,2)
        g = torch.tensor([2**-0.5,-2**-0.5], device=x.device, dtype=x.dtype).view(1,1,2)
        y = x.unsqueeze(1)
        A = F.conv1d(y, h, stride=2).squeeze(1)   # 低频
        Dw= F.conv1d(y, g, stride=2).squeeze(1)   # 高频
        if thr is not None:
            Dw = torch.where(Dw.abs() > thr, Dw, torch.zeros_like(Dw))
        # 逆变换
        A_up = F.conv_transpose1d(A.unsqueeze(1), h, stride=2)
        D_up = F.conv_transpose1d(Dw.unsqueeze(1), g, stride=2)
        rec = (A_up + D_up).squeeze(1)
        return rec[:, :x.size(1)]  # 裁回原长

    def _cheby(self, Z, L_norm, theta):
        if (L_norm is None) or (theta is None):
            return Z
        K = len(theta)-1
        T0 = Z
        if K == 0:
            return theta[0]*T0
        T1 = torch.sparse.mm(L_norm, Z)
        out = theta[0]*T0 + theta[1]*T1
        for k in range(2, K+1):
            T2 = 2*torch.sparse.mm(L_norm, T1) - T0
            out = out + theta[k]*T2
            T0, T1 = T1, T2
        return out


class LossScheduler:
    """
    调度 β、以及 rank/cl/edge/spec/tv 的权重；与 pre_epoch_processing 协同。
    """
    def __init__(self, total_epochs=200,
                 beta_min=0.05, beta_max=0.30, warmup=10, hold=20, decay=60, cosine=True,
                 w_rank=1.0, w_cl=1.0, w_edge=1.0, w_spec=0.0, w_tv=0.0):
        self.E = total_epochs
        self.beta_min, self.beta_max = beta_min, beta_max
        self.warmup, self.hold, self.decay, self.cosine = warmup, hold, decay, cosine
        self.base = dict(rank=w_rank, cl=w_cl, edge=w_edge, spec=w_spec, tv=w_tv)

    def beta_at(self, e):
        if e <= self.warmup:
            t = e / max(1, self.warmup)
            return self.beta_min + (self.beta_max - self.beta_min)*t
        elif e <= self.warmup + self.hold:
            return self.beta_max
        else:
            T = max(1, self.decay)
            t = min(1.0, (e - self.warmup - self.hold)/T)
            if self.cosine:
                return self.beta_min + 0.5*(self.beta_max - self.beta_min)*(1+math.cos(math.pi*t))
            else:
                return self.beta_max - (self.beta_max - self.beta_min)*t

    def weights_at(self, e):
        # 一个简单策略：前期拉高 edge，中期稳，后期降；CL 余弦退火
        lam_edge = self.base['edge'] * (1.0 if e <= self.warmup + self.hold else 0.5)
        lam_cl   = self.base['cl'] * 0.5*(1 + math.cos(math.pi*min(1.0, max(0.0,(e-10)/max(1,self.decay)))))
        lam_rank = self.base['rank']
        lam_spec = self.base['spec'] * (1.0 if e >= 20 else 0.0)
        lam_tv   = self.base['tv'] * (1.0 if e >= 40 else 0.0)
        return dict(rank=lam_rank, cl=lam_cl, edge=lam_edge, spec=lam_spec, tv=lam_tv)


# =========================
# 新模型：SMOREDiffX
# =========================
import numpy as np
import torch
import torch.nn.functional as F
from models.smorediff import SMOREDiff

class SMOREDiffX(SMOREDiff):
    """
    在 SMOREDiff 基础上集成：
      - NoiseRecipe：边重建采样（合成假阳性）
      - DSPDenoiser：条件表征前的可插拔降噪
      - LossScheduler：β与loss权重调度
    """
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        # === 模块开关 ===
        self.noise_recipe_on = bool(config.get('noise_recipe_on', True))
        self.dsp_mode        = config.get('dsp_mode', 'fft_mask')  # ['fft_mask','wavelet','cheby','none']
        self.fft_keep_ratio  = float(config.get('fft_keep_ratio', 0.5))
        self.wavelet_thr     = config.get('wavelet_thr', None)
        self.loss_sched_on   = bool(config.get('loss_sched_on', True))
        self.total_epochs    = int(config.get('epochs', 200))

        # NoiseRecipe
        self.nr = NoiseRecipe(self.n_users, self.n_items,
                              topk_fake=int(config.get('nr_topk_fake', 5)),
                              drop_rate_hi=float(config.get('nr_drop_hi', 0.1)),
                              drop_rate_lo=float(config.get('nr_drop_lo', 0.02)))

        # DSP
        self.dsp = DSPDenoiser(
            mode=self.dsp_mode,
            fft_keep_ratio=self.fft_keep_ratio,
            wavelet_thr=self.wavelet_thr,
            cheby_theta=None, cheby_L=None  # 需要时可传入
        )

        # LossScheduler
        self.sched = LossScheduler(
            total_epochs=self.total_epochs,
            beta_min=float(config.get('beta_min', 0.05)),
            beta_max=float(config.get('beta_max', 0.30)),
            warmup=int(config.get('beta_warmup_ep', 10)),
            hold=int(config.get('beta_hold_ep', 20)),
            decay=int(config.get('beta_decay_ep', 60)),
            cosine=bool(config.get('use_cosine_decay', True)),
            w_rank=float(config.get('w_rank', 1.0)),
            w_cl=float(config.get('w_cl', 1.0)),
            w_edge=float(config.get('w_edge', 1.0)),
            w_spec=float(config.get('w_spec', 0.0)),
            w_tv=float(config.get('w_tv', 0.0)),
        )

        # EMA + 行和守恒（来自之前建议）
        self.use_softR_ema   = bool(config.get('use_softR_ema', True))
        self.softR_ema_tau   = float(config.get('softR_ema_tau', 0.2))
        self._softR_ema      = None
        self.renorm_softR    = bool(config.get('renorm_softR', True))

        # 记录 epoch
        self._epoch_ptr = 0

    def pre_epoch_processing(self):
        # 调度 β 与“仅下调”阶段
        self._epoch_ptr += 1
        e = self._epoch_ptr
        if self.loss_sched_on:
            self.diff_beta = self.sched.beta_at(e)
            # warmup 階段只下调，之后允许上调
            self.diff_only_down = (e <= int(getattr(self.sched, 'warmup', 10)))

    # 覆盖：对 cond_item 加 DSP 降噪 + soft_R 的 EMA 与行和守恒
    def _build_soft_R_once(self, user_id_embeds: torch.Tensor, cond_item_embeds: torch.Tensor):
        # 先做 DSP 降噪（可关）
        cond_item_embeds = self.dsp(cond_item_embeds) if self.dsp_mode != 'none' else cond_item_embeds

        soft_R = super()._build_soft_R_once(user_id_embeds, cond_item_embeds)

        # 行和守恒
        if self.renorm_softR:
            R_coo = self.R.coalesce()
            u_idx = R_coo.indices()[0]
            orig_row_sum = torch.zeros(self.n_users, device=soft_R.device).index_add_(0, u_idx, R_coo.values())
            new_row_sum  = torch.zeros(self.n_users, device=soft_R.device).index_add_(0, u_idx, soft_R.values()).clamp_min(1e-6)
            scale = (orig_row_sum / new_row_sum)[u_idx]
            new_val = soft_R.values() * scale
            soft_R = torch.sparse_coo_tensor(soft_R.indices(), new_val, soft_R.size(), device=soft_R.device).coalesce()

        # EMA
        if self.use_softR_ema:
            if self._softR_ema is None:
                self._softR_ema = soft_R
            else:
                v_ema = self._softR_ema.values()
                v_cur = soft_R.values()
                v_new = (1 - self.softR_ema_tau) * v_ema + self.softR_ema_tau * v_cur
                self._softR_ema = torch.sparse_coo_tensor(soft_R.indices(), v_new, soft_R.size(), device=soft_R.device).coalesce()
            soft_R = self._softR_ema

        return soft_R

    # 额外的边重建损失（PU 风格）
    def _edge_recon_loss(self, users, pos_items, fusion_item_embeds):
        if not self.noise_recipe_on:
            return torch.tensor(0.0, device=fusion_item_embeds.device)

        # 构造正/假阳性对
        edge_pos_u, edge_pos_i, edge_neg_u, edge_neg_i = self.nr._edge_recon_pairs(
            users, pos_items, None, fusion_item_embeds, self.R, k_hard=int(getattr(self.nr, 'topk_fake', 5))
        )
        # 打分（与 soft_R 的打分同源：余弦 + 温度）
        u_pos = F.normalize(self.user_embedding.weight[edge_pos_u], dim=-1)
        i_pos = F.normalize(fusion_item_embeds[edge_pos_i], dim=-1)
        u_neg = F.normalize(self.user_embedding.weight[edge_neg_u], dim=-1)
        i_neg = F.normalize(fusion_item_embeds[edge_neg_i], dim=-1)

        s_pos = (u_pos * i_pos).sum(-1) / max(self.diff_temp, 1e-6)
        s_neg = (u_neg * i_neg).sum(-1) / max(self.diff_temp, 1e-6)

        # PU 风格：正例=1，合成假阳性=0，未观测其余不计
        y_pos = torch.ones_like(s_pos)
        y_neg = torch.zeros_like(s_neg)
        bce = F.binary_cross_entropy_with_logits
        loss = bce(s_pos, y_pos) + bce(s_neg, y_neg)
        return loss

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(self.norm_adj, train=True)

        u_g = ua_embeddings[users]
        pi_g = ia_embeddings[pos_items]
        ni_g = ia_embeddings[neg_items]

        # 1) 排序损失
        mf_loss, emb_loss, reg_loss = self.bpr_loss(u_g, pi_g, ni_g)

        # 2) CL（与原版一致）
        side_u, side_i = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        cont_u, cont_i = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl = self.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + self.InfoNCE(side_u[users], cont_u[users], 0.2)

        # 3) 边重建损失（PU）
        # 使用“未传播前”的融合 item 表征作为条件
        # 复用 forward 里 cond_item 的定义：这里简单从 item_id_embedding 侧重建一遍
        # 更严格可在 forward 保存 cond_item；为最小入侵，这里用 text/image 投影再门控
        if (self.v_feat is not None) and (self.t_feat is not None):
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats  = self.text_trs(self.text_embedding.weight)
            _, _, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
            cond_item = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))
        else:
            cond_item = self.item_id_embedding.weight  # 回退

        edge_recon = self._edge_recon_loss(users, pos_items, cond_item)

        # 4) 权重调度
        if self.loss_sched_on:
            ws = self.sched.weights_at(self._epoch_ptr or 1)
            loss = (ws['rank'] * (mf_loss + emb_loss + reg_loss)
                    + ws['cl'] * cl
                    + ws['edge'] * edge_recon)
        else:
            loss = (mf_loss + emb_loss + reg_loss + self.cl_loss * cl + 1.0 * edge_recon)

        return loss
