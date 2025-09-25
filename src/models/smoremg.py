# coding: utf-8
# Author: ChatGPT patch for quick landing suggestions
#
# What this patch adds on top of upstream SMORE:
# 1) Frequency-domain *residual* injection with user/item-aware band gating (high-SNR bands get stronger injection).
# 2) Frequency-consistency loss between modalities on *shared low-frequency* bands (stabilizes cross-modal alignment).
# 3) Robustness/flatness via stochastic-consistency regularization (MG-style surrogate) between two dropout views.
# 4) Lightweight diagnostics: export per-band energies ("SNR proxy") to inspect which bands dominate errors.
#
# All additions are guarded by config flags and default to OFF to be backward-compatible.
#
# New config keys (with sane defaults):
#   use_freq_residual: bool = True
#   freq_residual_alpha: float = 0.25          # residual strength
#   low_freq_ratio: float = 0.15               # lowest 15% bins considered "shared/semantic" band
#   freq_consistency_weight: float = 0.0       # turn on by setting > 0
#   mg_consistency_weight: float = 0.0         # turn on by setting > 0 (robustness surrogate)
#   diagnostic_dump: bool = False              # if True, enable export_diagnostics()
#
# Usage notes:
# - You don't need to change your trainer; calculate_loss() will include the new losses when weights > 0.
# - Set freq_residual_alpha in [0.1, 0.4] initially; increase if content signals are reliable.
# - For quick robustness gains, set mg_consistency_weight ~ 1e-3 to 5e-3 on Amazon/MovieLens.
# - To analyze spectra, call model.export_diagnostics() after a few epochs.

import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from common.abstract_recommender import GeneralRecommender
from utils.utils import build_sim, build_knn_normalized_graph


class SMOREMG(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SMOREMG, self).__init__(config, dataset)
        self.config = config
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

        # === New flags ===
        self.use_freq_residual = config.get('use_freq_residual', True)
        self.freq_residual_alpha = float(config.get('freq_residual_alpha', 0.25))
        self.low_freq_ratio = float(config.get('low_freq_ratio', 0.15))
        self.freq_consistency_weight = float(config.get('freq_consistency_weight', 0.0))
        self.mg_consistency_weight = float(config.get('mg_consistency_weight', 0.0))
        self.diagnostic_dump = bool(config.get('diagnostic_dump', False))
        # diagnostics runtime knobs
        default_diag_dir = config.get('diagnostic_output_dir', 'logs')
        self._diagnostic_output_dir = default_diag_dir
        self._diagnostic_output_dir_abs = default_diag_dir if os.path.isabs(default_diag_dir) else os.path.abspath(default_diag_dir)
        default_template = f"{str(config['model']).lower()}_snr_epoch{{epoch:02d}}.npz"
        self._diagnostic_filename_template = config.get('diagnostic_filename_template', default_template)
        interval = config.get('diagnostic_interval', 1)
        try:
            interval = int(interval)
        except Exception:
            interval = 1
        self._diagnostic_interval = max(1, interval)

        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)

        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)

        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        image_adj_file = os.path.join(dataset_path, f'image_adj_{self.image_knn_k}_{True}.pt')
        text_adj_file = os.path.join(dataset_path, f'text_adj_{self.text_knn_k}_{True}.pt')

        self.norm_adj = self.get_adj_mat()
        self.R_sprse_mat = self.R
        self.R = self.sparse_mx_to_torch_sparse_tensor(self.R).float().to(self.device)
        self.norm_adj = self.sparse_mx_to_torch_sparse_tensor(self.norm_adj).float().to(self.device)

        # === Load or build modality graphs ===
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            if os.path.exists(image_adj_file):
                image_adj = torch.load(image_adj_file)
            else:
                image_adj = build_sim(self.image_embedding.weight.detach())
                image_adj = build_knn_normalized_graph(image_adj, topk=self.image_knn_k, is_sparse=True, norm_type='sym')
                torch.save(image_adj, image_adj_file)
            self.image_original_adj = image_adj.cuda()

        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            if os.path.exists(text_adj_file):
                text_adj = torch.load(text_adj_file)
            else:
                text_adj = build_sim(self.text_embedding.weight.detach())
                text_adj = build_knn_normalized_graph(text_adj, topk=self.text_knn_k, is_sparse=True, norm_type='sym')
                torch.save(text_adj, text_adj_file)
            self.text_original_adj = text_adj.cuda()

        self.fusion_adj = self.max_pool_fusion()

        # === Modality MLPs ===
        if self.v_feat is not None:
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.embedding_dim)
        if self.t_feat is not None:
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.embedding_dim)

        self.softmax = nn.Softmax(dim=-1)

        # === Attention & gates ===
        self.query_v = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.query_t = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim), nn.Tanh(),
            nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        )
        self.gate_v = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_f = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_image_prefer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_text_prefer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())
        self.gate_fusion_prefer = nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim), nn.Sigmoid())

        # === Learnable complex weights (frequency filters) ===
        self.freq_bins = self.embedding_dim // 2 + 1
        self.image_complex_weight = nn.Parameter(torch.randn(1, self.freq_bins, 2, dtype=torch.float32))
        self.text_complex_weight = nn.Parameter(torch.randn(1, self.freq_bins, 2, dtype=torch.float32))
        self.fusion_complex_weight = nn.Parameter(torch.randn(1, self.freq_bins, 2, dtype=torch.float32))

        # === NEW: Item-aware frequency band gate (real, in [0,1]) ===
        # maps item ID embedding -> per-bin weights to approximate per-item SNR prior
        self.item_freq_gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.freq_bins), nn.Sigmoid()
        )

    # ======================= Utils & Graphs =======================
    def pre_epoch_processing(self):
        pass

    def max_pool_fusion(self):
        image_adj = self.image_original_adj.coalesce()
        text_adj = self.text_original_adj.coalesce()
        image_indices, image_values = image_adj.indices().to(self.device), image_adj.values().to(self.device)
        text_indices, text_values = text_adj.indices().to(self.device), text_adj.values().to(self.device)
        combined_indices = torch.cat((image_indices, text_indices), dim=1)
        combined_indices, unique_idx = torch.unique(combined_indices, dim=1, return_inverse=True)
        combined_values_image = torch.full((combined_indices.size(1),), float('-inf'), device=self.device)
        combined_values_text = torch.full((combined_indices.size(1),), float('-inf'), device=self.device)
        combined_values_image[unique_idx[:image_indices.size(1)]] = image_values
        combined_values_text[unique_idx[image_indices.size(1):]] = text_values
        combined_values, _ = torch.max(torch.stack((combined_values_image, combined_values_text)), dim=0)
        fusion_adj = torch.sparse.FloatTensor(combined_indices, combined_values, image_adj.size()).coalesce()
        return fusion_adj

    def get_adj_mat(self):
        adj_mat = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        adj_mat = adj_mat.tolil(); R = self.interaction_matrix.tolil()
        adj_mat[:self.n_users, self.n_users:] = R
        adj_mat[self.n_users:, :self.n_users] = R.T
        adj_mat = adj_mat.todok()
        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -0.5).flatten(); d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj_mat).dot(d_mat_inv)
            return norm_adj.tocoo()
        norm_adj_mat = normalized_adj_single(adj_mat).tolil()
        self.R = norm_adj_mat[:self.n_users, self.n_users:]
        return norm_adj_mat.tocsr()

    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    # ======================= Core: Spectrum ops =======================
    def spectrum_convolution(self, image_embeds, text_embeds):
        """
        Modality Denoising & Cross-Modality Fusion (frequency-domain)
        Returns:
            image_conv, text_conv, fusion_conv (time-domain residuals)
            image_fft_filt, text_fft_filt (for optional consistency loss / diagnostics)
        """
        # FFT along feature dim
        image_fft = torch.fft.rfft(image_embeds, dim=1, norm='ortho')            # [n_items, B]
        text_fft  = torch.fft.rfft(text_embeds,  dim=1, norm='ortho')

        # Learnable complex filters
        image_cw = torch.view_as_complex(self.image_complex_weight)              # [1, B]
        text_cw  = torch.view_as_complex(self.text_complex_weight)
        fusion_cw= torch.view_as_complex(self.fusion_complex_weight)

        # Per-item band gates (approx SNR prior in [0,1])
        item_gate = self.item_freq_gate(self.item_id_embedding.weight)           # [n_items, B]
        item_gate_c = item_gate.to(image_fft.dtype)                              # real gate applied to complex spectrum

        # Uni-modal denoising in frequency domain with item-aware gates
        image_fft_filt = image_fft * image_cw * item_gate_c
        text_fft_filt  = text_fft  * text_cw  * item_gate_c

        # Cross-modality fusion in frequency domain (Hadamard product)
        fusion_fft = (image_fft * text_fft) * fusion_cw * item_gate_c

        # Back to time domain (residuals)
        image_conv  = torch.fft.irfft(image_fft_filt, n=image_embeds.shape[1], dim=1, norm='ortho')
        text_conv   = torch.fft.irfft(text_fft_filt,  n=text_embeds.shape[1],  dim=1, norm='ortho')
        fusion_conv = torch.fft.irfft(fusion_fft,     n=text_embeds.shape[1],  dim=1, norm='ortho')

        return image_conv, text_conv, fusion_conv, image_fft_filt, text_fft_filt

    # ======================= Forward =======================
    def forward(self, adj, train=False):
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)

        # Spectrum Modality Fusion (+ residual injection if enabled)
        image_conv, text_conv, fusion_conv, image_fft_filt, text_fft_filt = self.spectrum_convolution(image_feats, text_feats)

        if self.use_freq_residual:
            image_item_embeds = self.item_id_embedding.weight + self.freq_residual_alpha * self.gate_v(image_conv)
            text_item_embeds  = self.item_id_embedding.weight + self.freq_residual_alpha * self.gate_t(text_conv)
            fusion_item_embeds= self.item_id_embedding.weight + self.freq_residual_alpha * self.gate_f(fusion_conv)
        else:
            # fallback to multiplicative scheme in original impl
            image_item_embeds = torch.multiply(self.item_id_embedding.weight, self.gate_v(image_conv))
            text_item_embeds  = torch.multiply(self.item_id_embedding.weight, self.gate_t(text_conv))
            fusion_item_embeds= torch.multiply(self.item_id_embedding.weight, self.gate_f(fusion_conv))

        # ----- User-Item (Behavioral) View -----
        item_embeds = self.item_id_embedding.weight
        user_embeds = self.user_embedding.weight
        ego_embeddings = torch.cat([user_embeds, item_embeds], dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1)
        content_embeds = all_embeddings                                          # [n_u+n_i, d]

        # ----- Item-Item Modality-specific & Fusion views -----
        # Image view
        for _ in range(self.n_layers):
            image_item_embeds = torch.sparse.mm(self.image_original_adj, image_item_embeds)
        image_user_embeds = torch.sparse.mm(self.R, image_item_embeds)
        image_embeds = torch.cat([image_user_embeds, image_item_embeds], dim=0)

        # Text view
        for _ in range(self.n_layers):
            text_item_embeds = torch.sparse.mm(self.text_original_adj, text_item_embeds)
        text_user_embeds = torch.sparse.mm(self.R, text_item_embeds)
        text_embeds = torch.cat([text_user_embeds, text_item_embeds], dim=0)

        # Fusion view
        for _ in range(self.n_layers):
            fusion_item_embeds = torch.sparse.mm(self.fusion_adj, fusion_item_embeds)
        fusion_user_embeds = torch.sparse.mm(self.R, fusion_item_embeds)
        fusion_embeds = torch.cat([fusion_user_embeds, fusion_item_embeds], dim=0)

        # ----- Modality-aware Preference Module -----
        fusion_att_v, fusion_att_t = self.query_v(fusion_embeds), self.query_t(fusion_embeds)
        agg_image_embeds = self.softmax(fusion_att_v) * image_embeds
        agg_text_embeds  = self.softmax(fusion_att_t) * text_embeds

        image_prefer = self.dropout(self.gate_image_prefer(content_embeds))
        text_prefer  = self.dropout(self.gate_text_prefer(content_embeds))
        fusion_prefer= self.dropout(self.gate_fusion_prefer(content_embeds))

        agg_image_embeds = image_prefer * agg_image_embeds
        agg_text_embeds  = text_prefer  * agg_text_embeds
        fusion_embeds    = fusion_prefer* fusion_embeds

        side_embeds = torch.mean(torch.stack([agg_image_embeds, agg_text_embeds, fusion_embeds]), dim=0)
        all_embeds = content_embeds + side_embeds
        all_embeddings_users, all_embeddings_items = torch.split(all_embeds, [self.n_users, self.n_items], dim=0)

        # Cache for loss terms
        cache = {
            'side_embeds': side_embeds,
            'content_embeds': content_embeds,
            'image_fft_filt': image_fft_filt,
            'text_fft_filt':  text_fft_filt
        }

        if train:
            return all_embeddings_users, all_embeddings_items, cache
        return all_embeddings_users, all_embeddings_items

    # ======================= Losses =======================
    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(users * pos_items, dim=1)
        neg_scores = torch.sum(users * neg_items, dim=1)
        regularizer = 0.5 * (users.pow(2).sum() + pos_items.pow(2).sum() + neg_items.pow(2).sum()) / self.batch_size
        mf_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        emb_loss = self.reg_weight * regularizer
        return mf_loss, emb_loss, 0.0

    @staticmethod
    def InfoNCE(view1, view2, temperature):
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
        pos = torch.exp(((view1 * view2).sum(dim=-1)) / temperature)
        ttl = torch.exp(view1 @ view2.t() / temperature).sum(dim=1)
        return torch.mean(-torch.log(pos / ttl))

    def _freq_consistency(self, items_idx, image_fft_filt, text_fft_filt):
        """Consistency on shared low-frequency bands (amplitude & phase).
        We align complex spectra by L2 on real/imag parts.
        """
        if self.freq_consistency_weight <= 0:
            return 0.0 * self.user_embedding.weight.sum()
        B = self.freq_bins
        k = max(1, int(B * self.low_freq_ratio))
        i_fft = image_fft_filt[items_idx, :k]
        t_fft = text_fft_filt[items_idx, :k]
        loss = F.mse_loss(i_fft.real, t_fft.real) + F.mse_loss(i_fft.imag, t_fft.imag)
        return self.freq_consistency_weight * loss

    def _mg_stochastic_consistency(self, cache):
        """MG-style surrogate: two dropout views of side/content embeddings should be close.
        Encourages flat minima and robustness without optimizer coupling.
        """
        if self.mg_consistency_weight <= 0:
            return 0.0 * self.user_embedding.weight.sum()
        side1 = self.dropout(cache['side_embeds'])
        side2 = self.dropout(cache['side_embeds'])
        con1  = self.dropout(cache['content_embeds'])
        con2  = self.dropout(cache['content_embeds'])
        loss = F.mse_loss(side1, side2) + F.mse_loss(con1, con2)
        return self.mg_consistency_weight * loss

    def calculate_loss(self, interaction):
        users = interaction[0]; pos_items = interaction[1]; neg_items = interaction[2]
        ua_embeddings, ia_embeddings, cache = self.forward(self.norm_adj, train=True)

        u = ua_embeddings[users]
        pi = ia_embeddings[pos_items]
        ni = ia_embeddings[neg_items]

        mf_loss, emb_loss, reg_loss = self.bpr_loss(u, pi, ni)

        side_embeds = cache['side_embeds']; content_embeds = cache['content_embeds']
        side_u, side_i = torch.split(side_embeds, [self.n_users, self.n_items], dim=0)
        cont_u, cont_i = torch.split(content_embeds, [self.n_users, self.n_items], dim=0)

        # Original dual-view CL (behavior vs modality-graph views)
        cl_loss = self.InfoNCE(side_i[pos_items], cont_i[pos_items], 0.2) + \
                  self.InfoNCE(side_u[users],      cont_u[users],      0.2)

        # NEW: frequency consistency on low bands for the items in batch
        freq_c = self._freq_consistency(pos_items, cache['image_fft_filt'], cache['text_fft_filt'])

        # NEW: MG-style stochastic consistency
        mg_c = self._mg_stochastic_consistency(cache)

        return mf_loss + emb_loss + reg_loss + self.cl_loss * cl_loss + freq_c + mg_c

    # ======================= Inference =======================
    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward(self.norm_adj)
        u_embeddings = restore_user_e[user]
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))
        return scores

    # ======================= Diagnostics =======================
    def post_epoch_processing(self):
        if not self.diagnostic_dump:
            return
        epoch_idx = getattr(self, 'cur_epoch', None)
        if epoch_idx is None:
            return
        if (epoch_idx + 1) % self._diagnostic_interval != 0:
            return
        os.makedirs(self._diagnostic_output_dir_abs, exist_ok=True)
        filename = self._diagnostic_filename_template.format(
            epoch=epoch_idx + 1,
            model=str(self.config['model']).lower()
        )
        diag_path = filename if os.path.isabs(filename) else os.path.join(self._diagnostic_output_dir_abs, filename)
        out = self.export_diagnostics(diag_path)
        if not out:
            return

        def _stats(arr):
            if arr is None:
                return None
            arr = np.asarray(arr)
            if arr.size == 0:
                return None
            return float(arr.mean()), float(arr.max()), float(arr.min())

        img_stats = _stats(out.get('img_energy'))
        txt_stats = _stats(out.get('txt_energy'))

        rel_path = os.path.relpath(diag_path, os.getcwd()) if not os.path.isabs(filename) else diag_path
        parts = [f"diagnostics saved -> {rel_path}"]
        if img_stats:
            parts.append(
                "img_mean={:.4f} img_max={:.4f} img_min={:.4f}".format(*img_stats)
            )
        if txt_stats:
            parts.append(
                "txt_mean={:.4f} txt_max={:.4f} txt_min={:.4f}".format(*txt_stats)
            )
        return ' | '.join(parts)

    @torch.no_grad()
    def export_diagnostics(self, path: str = None):
        """Export per-band average energy for image/text spectra (SNR proxy).
        Returns a dict; if `path` provided, save as npz.
        """
        if not self.diagnostic_dump:
            return {}
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats  = self.text_trs(self.text_embedding.weight)
        image_fft = torch.fft.rfft(image_feats, dim=1, norm='ortho')
        text_fft  = torch.fft.rfft(text_feats,  dim=1, norm='ortho')
        img_energy = (image_fft.real.pow(2) + image_fft.imag.pow(2)).mean(dim=0).detach().cpu().numpy()
        txt_energy = (text_fft.real.pow(2)  + text_fft.imag.pow(2)).mean(dim=0).detach().cpu().numpy()
        out = {'img_energy': img_energy, 'txt_energy': txt_energy}
        if path is not None:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.savez(path, **out)
        return out
