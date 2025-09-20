# coding: utf-8
# rongqing001@e.ntu.edu.sg
r"""
SMORE - Multi-modal Recommender System
Reference:
    ACM WSDM 2025: Spectrum-based Modality Representation Fusion Graph Convolutional Network for Multimodal Recommendation

Reference Code:
    https://github.com/kennethorq/SMORE
"""

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, compute_normalized_laplacian, build_knn_neighbourhood, build_knn_normalized_graph


class SMOREDiff(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMOREDiff, self).__init__(config, dataset)
        self.sparse = True
        self.cl_loss = config['cl_loss']
        self.n_ui_layers = config['n_ui_layers']
        self.embedding_dim = config['embedding_size']
        self.n_layers = config['n_layers']
        self.reg_weight = config['reg_weight']
        self.image_knn_k = config['image_knn_k']
        self.text_knn_k = config['text_knn_k']
        self.dropout_rate = config['dropout_rate']
        self.dropout = nn.Dropout(p=self.dropout_rate)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, 'image_adj_{}_{}.pt'.format(self.image_knn_k, self.sparse))
        text_adj_file = os.path.join(dataset_path, 'text_adj_{}_{}.pt'.format(self.text_knn_k, self.sparse))

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=self.sparse,
                                                       norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=self.sparse, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda() 

        self.fusion_adj = self.max_pool_fusion()

        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )

        self.gate_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_f = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_image_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.gate_text_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )
        self.gate_fusion_prefer = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Sigmoid()
        )

        self.image_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.embedding_dim // 2 + 1, 2, dtype=torch.float32))
        self.use_diffusion_mvp = bool(config.get('use_diffusion_mvp', False))
        self.diff_beta = float(config.get('diff_beta', 0.5))     # 建议 0.05~0.2 起步
        self.diff_temp   = float(config.get('diff_temp', 0.6))      # 余弦温度 T
        self.diff_gamma  = float(config.get('diff_gamma', 1.2))     # tanh 强度 γ
        self.diff_z_thres= float(config.get('diff_z_thres', 0.5))   # z-score 阈 τ
        self.diff_only_down = bool(config.get('diff_only_down', True))
        self.diff_stopgrad  = bool(config.get('diff_stopgrad', True))    # 断开条件梯度
        self.spec_method = str(config.get('spec_method', 'fft')).lower()  # 'fft' | 'dwt' | 'wiener'

        # DWT 小波相关
        self.dwt_keep_ratio = float(config.get('dwt_keep_ratio', 0.5))    # 仅保留D子带Top-p的系数
        self.dwt_use_softthr = bool(config.get('dwt_use_softthr', False)) # 是否对D子带做Soft阈值
        self.dwt_softthr_k = float(config.get('dwt_softthr_k', 0.0))      # Soft阈值比例(对每个item的|D|分布设阈)

        # Wiener 收缩相关
        self.wiener_alpha = float(config.get('wiener_alpha', 0.5))        # 噪声方差估计的比例（用全局中位数）
        self.wiener_eps   = float(config.get('wiener_eps', 1e-6))

        # edge_nce loss 相关
        self.use_edge_nce   = bool(config.get('use_edge_nce', True))  # 开关
        self.edge_nce_k     = int(config.get('edge_nce_k', 5))        # 每个正样本的硬负例数
        self.edge_nce_temp  = float(config.get('edge_nce_temp', 0.6)) # 温度T
        self.edge_nce_lambda= float(config.get('edge_nce_lambda', 0.2))# 权重
        self.edge_use_user_pool = bool(config.get('edge_use_user_pool', True))
        self.edge_user_pool_L0  = int(config.get('edge_user_pool_L0', 50))  # 每个正例取前L0邻居后合并
        self.edge_inbatch_negs  = bool(config.get('edge_inbatch_negs', True))
        self.edge_inbatch_ratio = int(config.get('edge_inbatch_ratio', 2))

        R_csr = self.R_sprse_mat.tocsr()   # 这是你在 __init__ 里保存的 scipy 稀疏 R
        self.user_pos_items = [set(R_csr[u].indices) for u in range(self.n_users)]

        fadj = self.fusion_adj.coalesce().cpu()
        fi, fj = fadj.indices()
        fv = fadj.values()
        neighbors = [[] for _ in range(self.n_items)]
        for a, b, w in zip(fi.tolist(), fj.tolist(), fv.tolist()):
            neighbors[a].append((b, w))
        for i in range(self.n_items):
            neighbors[i].sort(key=lambda x: x[1], reverse=True)
            neighbors[i] = [j for j,_ in neighbors[i]]
        self.fusion_knn = neighbors  # List[List[int]]

        self.user_candidate_pool = []
        L0 = self.edge_user_pool_L0
        for u in range(self.n_users):
            cands = set()
            for ip in self.user_pos_items[u]:
                for j in self.fusion_knn[ip][:L0]:
                    if j not in self.user_pos_items[u]:
                        cands.add(j)
            self.user_candidate_pool.append(list(cands))


    def _edge_logits(self, u_idx: torch.Tensor, i_idx: torch.Tensor, cond_item: torch.Tensor):
        # 余弦+温度，默认stop-grad与我们之前一致
        u = self.user_embedding.weight[u_idx]
        z = cond_item[i_idx]
        u = F.normalize(u.detach(), dim=-1)
        z = F.normalize(z.detach(), dim=-1)
        return (u * z).sum(dim=-1) / self.edge_nce_temp  # [B] or [B*(K+1)]

    def _edge_nce_loss(self, users, pos_items, cond_item):
        B, K = users.size(0), self.edge_nce_k
        device = users.device
        neg_list = []
        for u, ip in zip(users.tolist(), pos_items.tolist()):
            # 先从“用户候选池”取
            pool = self.user_candidate_pool[u] if self.edge_use_user_pool else []
            negs = []
            for j in pool:
                if len(negs) >= K: break
                negs.append(j)
            # 若不足则从“该正例的KNN”补齐
            if len(negs) < K:
                for j in self.fusion_knn[ip]:
                    if j not in self.user_pos_items[u] and j not in negs:
                        negs.append(j)
                        if len(negs) >= K: break
            # 再随机兜底
            while len(negs) < K:
                j = torch.randint(low=0, high=self.n_items, size=(1,)).item()
                if j not in self.user_pos_items[u] and j != ip and j not in negs:
                    negs.append(j)
            neg_list.append(negs[:K])

        # 组装 [B, 1+K]
        u_rep = users.view(-1,1).repeat(1, K+1).to(device)
        items = torch.zeros(B, K+1, dtype=torch.long, device=device); items[:,0] = pos_items
        for r, negs in enumerate(neg_list): items[r,1:1+K] = torch.tensor(negs, device=device)

        # —— in-batch 硬负例（可选）——
        if self.edge_inbatch_negs and B > 1:
            extra = min(self.edge_inbatch_ratio*K, B-1)
            # 简单取“同批其它样本的正例”
            ib_negs = pos_items[torch.randperm(B)[:extra]]
            # 拼到每一行后面（广播相同一组，简单且有效）
            items = torch.cat([items, ib_negs.view(1,-1).repeat(B,1)], dim=1)
            u_rep = users.view(-1,1).repeat(1, items.size(1)).to(device)

        logits = self._edge_logits(u_rep.reshape(-1), items.reshape(-1), cond_item).view(B, -1)
        labels = torch.zeros(B, dtype=torch.long, device=device)  # 正样本在列0
        return F.cross_entropy(logits, labels)




    # ---------- DWT (Haar 1-level) ----------
    def _haar_dwt_1level(self, x: torch.Tensor):
        # x: [N, D] -> returns A,D: [N, D//2]
        N, D = x.shape
        if D % 2 == 1:  # 维度为奇数，右侧补零
            x = F.pad(x, (0, 1), mode='constant', value=0.0)
            D = D + 1
        h = torch.tensor([1/np.sqrt(2),  1/np.sqrt(2)], device=x.device, dtype=x.dtype).view(1,1,2)
        g = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], device=x.device, dtype=x.dtype).view(1,1,2)
        y = x.unsqueeze(1)                        # [N,1,D]
        A = F.conv1d(y, h, stride=2).squeeze(1)   # 低频
        Dcoef = F.conv1d(y, g, stride=2).squeeze(1)# 高频
        return A, Dcoef

    def _haar_idwt_1level(self, A: torch.Tensor, Dcoef: torch.Tensor):
        # A,D: [N, D//2] -> [N, 2*(D//2)]
        hT = torch.tensor([1/np.sqrt(2),  1/np.sqrt(2)], device=A.device, dtype=A.dtype).view(1,1,2)
        gT = torch.tensor([1/np.sqrt(2), -1/np.sqrt(2)], device=A.device, dtype=A.dtype).view(1,1,2)
        A_up = F.conv_transpose1d(A.unsqueeze(1), hT, stride=2)
        D_up = F.conv_transpose1d(Dcoef.unsqueeze(1), gT, stride=2)
        x_rec = (A_up + D_up).squeeze(1)          # [N, 2*(D//2)]
        return x_rec

    def _wavelet_denoise(self, x: torch.Tensor):
        # 一层Haar：保留A子带，高频D子带只保留Top-p能量（或做soft阈值）
        A, Dcoef = self._haar_dwt_1level(x)              # [N, d2], [N, d2]
        if self.dwt_use_softthr and self.dwt_softthr_k > 0:
            # soft-threshold：对每个样本的D按|D|分布取分位阈
            k = max(1, int(Dcoef.size(1) * self.dwt_softthr_k))
            thr = torch.topk(Dcoef.abs(), k, dim=1).values[:, -1:]    # [N,1]
            Dcoef = torch.sign(Dcoef) * torch.relu(Dcoef.abs() - thr) # soft shrink
        else:
            # Top-p 掩膜
            k = max(1, int(Dcoef.size(1) * self.dwt_keep_ratio))
            topk = torch.topk(Dcoef.abs(), k, dim=1).values[:, -1:].repeat(1, Dcoef.size(1))
            mask = (Dcoef.abs() >= topk).to(Dcoef.dtype)
            Dcoef = Dcoef * mask
        x_rec = self._haar_idwt_1level(A, Dcoef)
        # 若原通道为奇数，裁回原长度
        if x_rec.size(1) > x.size(1):
            x_rec = x_rec[:, :x.size(1)]
        return x_rec

    # ---------- Wiener / James-Stein 收缩 ----------
    def _wiener_denoise(self, x: torch.Tensor):
        # x: [N, D]，以 batch 维估计每维方差；噪声方差取全局方差中位数 * alpha
        # 适合轻度噪声，几乎零额外开销
        # 训练时可与梯度共同传播
        var_feat = x.var(dim=0, unbiased=False)                          # [D]
        noise_var = var_feat.median() * self.wiener_alpha + self.wiener_eps
        shrink = (var_feat / (var_feat + noise_var)).clamp(0.0, 1.0)     # [D]
        return x * shrink  # 广播到 [N,D]

        
    def _build_soft_R_once(self, user_id_embeds: torch.Tensor, cond_item_embeds: torch.Tensor):
        """
        构造软边 \tilde R：对 self.R 的非零项做“乘法微扰”，带 per-user 校准与阈值。
        """
        R_coo = self.R.coalesce()
        idx = R_coo.indices()        # [2, nnz]
        val = R_coo.values()         # [nnz]
        u_idx, i_idx = idx[0], idx[1]

        # 1) 准备特征（默认 stop-grad + 余弦，降低投机增益）
        u_vec = user_id_embeds[u_idx]
        z_vec = cond_item_embeds[i_idx]
        if self.diff_stopgrad:
            u_vec = u_vec.detach()
            z_vec = z_vec.detach()
        u_vec = F.normalize(u_vec, dim=-1)
        z_vec = F.normalize(z_vec, dim=-1)

        # 2) 置信打分（温度缩放）
        sim = (u_vec * z_vec).sum(dim=-1).clamp(-1.0, 1.0)     # [-1,1]
        logits = sim / self.diff_temp                           # [nnz]

        # 3) per-user 零均值 + 方差归一
        n_users = self.n_users
        ones = torch.ones_like(logits)
        # sum & count
        sum_per_u = torch.zeros(n_users, device=logits.device).index_add_(0, u_idx, logits)
        cnt_per_u = torch.zeros(n_users, device=logits.device).index_add_(0, u_idx, ones).clamp_min(1.0)
        mu_u = (sum_per_u / cnt_per_u)[u_idx]

        sum2_per_u = torch.zeros(n_users, device=logits.device).index_add_(0, u_idx, logits * logits)
        var_u = (sum2_per_u / cnt_per_u - (sum_per_u / cnt_per_u)**2).clamp_min(0.0)
        std_u = torch.sqrt(var_u + 1e-6)[u_idx]

        zscore = (logits - mu_u) / (std_u + 1e-6)

        # 4) tanh 压缩 + 置信阈
        delta = torch.tanh(self.diff_gamma * zscore)            # [-1,1]
        if self.diff_only_down:
            # 仅下调：只允许负向（抑制不可信边），不做上调
            delta = torch.minimum(delta, torch.zeros_like(delta))
        else:
            pos_mask = (zscore > self.diff_z_thres).float()
            delta = torch.relu(delta) * pos_mask + torch.minimum(delta, torch.zeros_like(delta))

        # 5) 乘法微扰（近似保度），并裁剪为正
        scale = (1.0 + self.diff_beta * delta).clamp(min=0.05)  # 防止数值过小
        new_val = val * scale

        soft_R = torch.sparse_coo_tensor(idx, new_val, R_coo.size(), device=R_coo.device).coalesce()
        return soft_R


    def pre_epoch_processing(self):
        pass

    def max_pool_fusion(self):
        image_adj = self.image_original_adj.coalesce()
        text_adj = self.text_original_adj.coalesce()

        image_indices = image_adj.indices().to(self.device)
        image_values = image_adj.values().to(self.device)
        text_indices = text_adj.indices().to(self.device)
        text_values = text_adj.values().to(self.device)

        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)

        combined_values_image = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf')).to(self.device)

        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)

        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()

        return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        R = self.interaction_matrix.tolil()

        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))

            d_inv = np.power(rowsum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)

            norm_adj = d_mat_inv.dot(adj_mat)
            norm_adj = norm_adj.dot(d_mat_inv)
            return norm_adj.tocoo()

        norm_adj_mat = normalized_adj_single(adj_mat)
        norm_adj_mat = norm_adj_mat.tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def spectrum_convolution(self, image_embeds, text_embeds):
        """
        Modality Denoising & Cross-Modality Fusion (switchable)
        """
        method = getattr(self, 'spec_method', 'fft')
        if method == 'fft':
            image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
            text_fft  = torch.fft.rfft(text_embeds,  dim=1, norm='ortho')
            image_complex_weight = torch.view_as_complex(self.image_complex_weight)
            text_complex_weight  = torch.view_as_complex(self.text_complex_weight)
            fusion_complex_weight= torch.view_as_complex(self.fusion_complex_weight)
            image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')
            text_conv  = torch.fft.irfft(text_fft  * text_complex_weight,  n=text_embeds.shape[1],  dim=1, norm='ortho')
            fusion_conv= torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')
            return image_conv, text_conv, fusion_conv

        elif method == 'dwt':
            image_dn = self._wavelet_denoise(image_embeds)   # [N,D]
            text_dn  = self._wavelet_denoise(text_embeds)    # [N,D]
            fusion_dn= image_dn * text_dn                    # 逐维乘，简洁稳健
            return image_dn, text_dn, fusion_dn

        elif method == 'wiener':
            image_dn = self._wiener_denoise(image_embeds)
            text_dn  = self._wiener_denoise(text_embeds)
            fusion_dn= image_dn * text_dn
            return image_dn, text_dn, fusion_dn

        else:
            raise ValueError(f"Unknown spec_method={method}")

    
    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        #   Spectrum Modality Fusion（原样）
        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
        image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
        text_item_embeds  = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
        fusion_item_embeds= torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        # ### Diff-MVP: 构造条件软边 \tilde R（只在开启时）
        if self.use_diffusion_mvp:
            # 条件项：建议用“未传播前”的融合 item 表征（避免图上传播放大噪声）
            cond_item = fusion_item_embeds
            soft_R = self._build_soft_R_once(self.user_embedding.weight, cond_item)
        else:
            soft_R = self.R  # 退化为原 R

        #   User-Item (Behavioral) View（原样）
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Views + 用户侧聚合 —— 这里把 R 改为 \tilde R
        #   Image-view
        if self.sparse:
            for _ in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for _ in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(soft_R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for _ in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for _ in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(soft_R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for _ in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for _ in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(soft_R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        #   Modality-aware Preference Module（原样）
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v); fusion_soft_t = self.softmax(fusion_att_t)
        agg_image_embeds = fusion_soft_v * image_embeds
        agg_text_embeds  = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer  = self.gate_text_prefer(content_embeds)
        fusion_prefer= self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds  = torch.multiply(text_prefer,  agg_text_embeds)
        fusion_embeds    = torch.multiply(fusion_prefer, fusion_embeds)

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)
        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            return all_embeddings_users, all_embeddings_items, side_embeds, content_embeds
        return all_embeddings_users, all_embeddings_items

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)

        regularizer = 1. / 2 * (users ** 2).sum() + 1. / 2 * (pos_items ** 2).sum() + 1. / 2 * (neg_items ** 2).sum()
        regularizer = regularizer / self.batch_size

        maxi = F.logsigmoid(pos_scores - neg_scores)
        mf_loss = -torch.mean(maxi)

        emb_loss = self.reg_weight * regularizer
        reg_loss = 0.0
        return mf_loss, emb_loss, reg_loss

    def InfoNCE(self, view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos_score = (view1 * view2).sum(dim=-1)
        pos_score = torch.exp(pos_score / temperature)
        ttl_score = torch.matmul(view1, view2.transpose(0, 1))
        ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
        cl_loss = -torch.log(pos_score / ttl_score)
        return torch.mean(cl_loss)

    def calculate_loss(self, interaction):
        users     = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(self.norm_adj, train=True)

        # 传统 BPR
        u_g = ua_embeddings[users]
        pi_g= ia_embeddings[pos_items]
        ni_g= ia_embeddings[neg_items]
        bpr_loss, emb_loss, reg_loss = self.bpr_loss(u_g, pi_g, ni_g)

        # CL（你原有的）
        side_u, side_i = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        cont_u, cont_i = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)
        cl_loss = self.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                self.InfoNCE(side_u[users],     cont_u[users],     0.2)

        # ====== Edge-NCE：给扩散/软边一个与Recall一致的监督 ======
        edge_nce = torch.tensor(0.0, device=self.device)
        if self.use_edge_nce:
            # 复用一次“未传播的融合 item 表征”（与 forward 同步）
            image_feats = self.image_trs(self.image_embedding.weight)
            text_feats  = self.text_trs(self.text_embedding.weight)
            _, _, fusion_conv = self.spectrum_convolution(image_feats, text_feats)
            cond_item = torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))  # 未图上传播
            edge_nce = self._edge_nce_loss(users, pos_items, cond_item)

        return bpr_loss + emb_loss + reg_loss + self.cl_loss * cl_loss + self.edge_nce_lambda * edge_nce


    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores