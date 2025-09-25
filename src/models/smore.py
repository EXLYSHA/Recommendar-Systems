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


class SMORE(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMORE, self).__init__(config, dataset)
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

        self.mg_enable: bool = bool(config.get('mg_enable', True))
        self.mg_interval: int = int(config.get('mg_interval', 3))  # τ，论文推荐 ~3
        self.mg_alpha: float = float(config.get('mg_alpha', 0.5))  # 0<beta<alpha
        self.mg_beta: float = float(config.get('mg_beta', 0.2))
        self.mg_verbose: bool = bool(config.get('mg_verbose', True))
        self.global_step: int = 0  # 训练步计数器（供 MG interval 与诊断使用）

        # 注入策略 & 频域正则
        self.inject_mode = config.get('inject_mode', 'residual')  # 'mul' or 'residual'
        self.inject_scale = float(config.get('inject_scale', 0.7))
        self.spectral_weight_norm = bool(config.get('spectral_weight_norm', True))

        # CL 温度（覆写默认的 0.2）
        self.cl_temp = float(config.get('cl_temp', 0.2))

        # 诊断输出选项（保留）
        self.diag_spectrum = bool(config.get('diag_spectrum', True))
        self.diag_gate = bool(config.get('diag_gate', True))
        self.diag_grad = bool(config.get('diag_grad', True))

        

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
        Modality Denoising & Cross-Modality Fusion
        同时返回频谱能量用于诊断（低/中/高频段能量占比）。
        """
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')
        text_fft = torch.fft.rfft(text_embeds, dim=1, norm='ortho')

        image_complex_weight = torch.view_as_complex(self.image_complex_weight)
        text_complex_weight  = torch.view_as_complex(self.text_complex_weight)
        fusion_complex_weight= torch.view_as_complex(self.fusion_complex_weight)

        if self.spectral_weight_norm:
            # 单位幅值（相位保留），避免某些频带被任意放大/压扁
            def unit_mag(wc):
                mag = torch.abs(wc)
                wc = wc / (mag + 1e-8)
                return wc
            image_complex_weight = unit_mag(image_complex_weight)
            text_complex_weight  = unit_mag(text_complex_weight)
            fusion_complex_weight= unit_mag(fusion_complex_weight)


        # Uni-modal Denoising
        image_conv = torch.fft.irfft(image_fft * image_complex_weight, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv = torch.fft.irfft(text_fft * text_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        # Cross-modality fusion
        fusion_conv = torch.fft.irfft(text_fft * image_fft * fusion_complex_weight, n=text_embeds.shape[1], dim=1, norm='ortho')

        # === 诊断：频带能量分布（幅度平方） ===
        with torch.no_grad():
            def band_energy(x_fft):  # x_fft: (N, F)
                mag2 = (x_fft.real**2 + x_fft.imag**2).mean(dim=0)  # 频带平均能量
                F = mag2.numel()
                lo = mag2[:max(1, F//3)].sum()
                mid = mag2[max(1, F//3):max(2, 2*F//3)].sum()
                hi = mag2[max(2, 2*F//3):].sum()
                total = lo + mid + hi + 1e-12
                return (lo/total).item(), (mid/total).item(), (hi/total).item()
            self._spec_energy_image = band_energy(image_fft)
            self._spec_energy_text  = band_energy(text_fft)

        return image_conv, text_conv, fusion_conv

    
    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        #   Spectrum Modality Fusion
# Spectrum Modality Fusion
        image_conv, text_conv, fusion_conv = self.spectrum_convolution(image_feats, text_feats)

        if self.inject_mode == 'mul':
            image_item_embeds = self.item_id_embedding.weight * self.gate_v(image_conv)
            text_item_embeds  = self.item_id_embedding.weight * self.gate_t(text_conv)
            fusion_item_embeds= self.item_id_embedding.weight * self.gate_f(fusion_conv)
        else:  # 'residual'（推荐）
            image_item_embeds = self.item_id_embedding.weight + self.inject_scale * self.gate_v(image_conv)
            text_item_embeds  = self.item_id_embedding.weight + self.inject_scale * self.gate_t(text_conv)
            fusion_item_embeds= self.item_id_embedding.weight + self.inject_scale * self.gate_f(fusion_conv)


        #   User-Item (Behavioral) View
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        content_embeds = all_embeddings

        #   Item-Item Modality Specific and Fusion views
        #   Image-view
        if self.sparse:
            for i in range(self.n_layers):
                image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        else:
            for i in range(self.n_layers):
                image_item_embeds = torch.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        #   Text-view
        if self.sparse:
            for i in range(self.n_layers):
                text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        else:
            for i in range(self.n_layers):
                text_item_embeds = torch.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        #   Fusion-view
        if self.sparse:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        else:
            for i in range(self.n_layers):
                fusion_item_embeds = torch.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        #   Modality-aware Preference Module
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        fusion_soft_v = self.softmax(fusion_att_v)
        agg_image_embeds = fusion_soft_v * image_embeds

        fusion_soft_t = self.softmax(fusion_att_t)
        agg_text_embeds = fusion_soft_t * text_embeds

        image_prefer = self.gate_image_prefer(content_embeds)
        text_prefer = self.gate_text_prefer(content_embeds)
        fusion_prefer = self.gate_fusion_prefer(content_embeds)
        image_prefer, text_prefer, fusion_prefer = self.dropout(image_prefer), self.dropout(text_prefer), self.dropout(fusion_prefer)

        agg_image_embeds = torch.multiply(image_prefer, agg_image_embeds)
        agg_text_embeds = torch.multiply(text_prefer, agg_text_embeds)
        fusion_embeds = torch.multiply(fusion_prefer, fusion_embeds)

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0) 

        all_embeds = content_embeds + side_embeds

        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        if train:
            if train and (self.mg_verbose or self.diag_gate):
                with torch.no_grad():
                    gv = torch.sigmoid(self.gate_v[0](image_conv))   # 取 Sigmoid 之前那层，再过 Sigmoid
                    gt = torch.sigmoid(self.gate_t[0](text_conv))
                    gf = torch.sigmoid(self.gate_f[0](fusion_conv))

                    def stats(x):
                        m = x.mean().item()
                        s = x.std().item()
                        sp = (x < 0.1).float().mean().item()  # 稀疏率：很小的门控比例
                        return m, s, sp
                    self._gate_act_stats = {
                        "gV(m/s/sp%)": stats(gv),
                        "gT(m/s/sp%)": stats(gt),
                        "gF(m/s/sp%)": stats(gf),
                    }


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
        users, pos_items, neg_items = interaction

        ua_embeddings, ia_embeddings, side_embeds, content_embeds = self.forward(self.norm_adj, train=True)
        self.global_step += 1

        u_g_embeddings = ua_embeddings[users]
        pos_i_g_embeddings = ia_embeddings[pos_items]
        neg_i_g_embeddings = ia_embeddings[neg_items]

        batch_mf_loss, batch_emb_loss, batch_reg_loss = self.bpr_loss(
            u_g_embeddings, pos_i_g_embeddings, neg_i_g_embeddings
        )

        side_embeds_users, side_embeds_items = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        content_embeds_user, content_embeds_items = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        cl_items = self.InfoNCE(side_embeds_items[pos_items], content_embeds_items[pos_items], self.cl_temp)
        cl_users = self.InfoNCE(side_embeds_users[users], content_embeds_user[users], self.cl_temp)
        cl_loss = cl_items + cl_users
        self._cl_stats = {"cl_items": cl_items.item(), "cl_users": cl_users.item()}

        return batch_mf_loss + batch_emb_loss + batch_reg_loss + self.cl_loss * cl_loss


    def full_sort_predict(self, interaction):
        user = interaction[0]

        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    @torch.no_grad()
    def log_mm_diagnostics(self, optimizer=None):
        if not (self.mg_verbose or self.diag_grad or self.diag_spectrum or self.diag_gate):
            return

        parts = []
        if self.diag_spectrum and hasattr(self, "_spec_energy_image"):
            img_lo, img_md, img_hi = self._spec_energy_image
            txt_lo, txt_md, txt_hi = self._spec_energy_text
            parts.append(f"[spec] image(lo/mid/hi)={img_lo:.2f}/{img_md:.2f}/{img_hi:.2f} "
                         f"text={txt_lo:.2f}/{txt_md:.2f}/{txt_hi:.2f}")
        if self.diag_gate and hasattr(self, "_gate_stats"):
            parts.append("[gate] " + ", ".join(f"{k}={v:.3f}" for k,v in self._gate_stats.items()))
        if hasattr(self, "_embed_norms"):
            parts.append("[emb] " + ", ".join(f"{k}={v:.2f}" for k,v in self._embed_norms.items()))
        if hasattr(self, "_cl_stats"):
            parts.append("[cl] " + ", ".join(f"{k}={v:.4f}" for k,v in self._cl_stats.items()))
        if hasattr(self, "_gate_act_stats"):
            parts.append("[gate_act] " + " ".join(f"{k}={v[0]:.3f}/{v[1]:.3f}/{100*v[2]:.1f}%" 
                                                for k, v in self._gate_act_stats.items()))

        if optimizer is not None and len(optimizer.param_groups)>0:
            lr = optimizer.param_groups[0].get("lr", float('nan'))
            parts.append(f"[mg] step={self.global_step} τ={self.mg_interval} α={self.mg_alpha} β={self.mg_beta} lr={lr}")

        print(" | ".join(parts))
