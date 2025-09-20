# -*- coding: utf-8 -*-
r"""
LightGCN
################################################

Reference:
    Xiangnan He et al. "LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation." in SIGIR 2020.

Reference code:
    https://github.com/kuandeng/LightGCN
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.abstract_recommender import GeneralRecommender
from common.loss import BPRLoss, EmbLoss
from common.init import xavier_uniform_initialization


class LightGCNCLIP(GeneralRecommender):
    r"""LightGCN is a GCN-based recommender model.

    LightGCN includes only the most essential component in GCN — neighborhood aggregation — for
    collaborative filtering. Specifically, LightGCN learns user and item embeddings by linearly
    propagating them on the user-item interaction graph, and uses the weighted sum of the embeddings
    learned at all layers as the final embedding.

    We implement the model following the original author with a pairwise training mode.
    """
    def __init__(self, config, dataset):
        super(LightGCNCLIP, self).__init__(config, dataset)

        # load dataset info
        self.interaction_matrix = dataset.inter_matrix(
            form='coo').astype(np.float32)

        # load parameters info
        self.latent_dim = config['embedding_size']  # int type:the embedding size of lightGCN
        self.n_layers = config['n_layers']  # int type:the layer num of lightGCN
        self.reg_weight = config['reg_weight']  # float32 type: the weight decay for l2 normalizaton

        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # projection dim for modalities; default to ID embedding size
        self.feat_embed_dim = config.get('feat_embed_dim', self.latent_dim)
        # lightweight fusion controls
        self.use_branch_norm = bool(config.get('use_branch_norm', True))
        self.use_degree_gate = bool(config.get('use_degree_gate', True))
        self.modal_drop_rate = float(config.get('modal_drop_rate', 0.0))
        # learnable residual weight for modality branch
        self.res_alpha = nn.Parameter(torch.tensor(float(config.get('res_alpha', 0.2)), dtype=torch.float32))
        # minimal floor to avoid total collapse during early training (optional)
        self.res_alpha_min = float(config.get('res_alpha_min', 0.0))
        # lightweight alignment loss to stabilize modality branch (optional, 0 disables)
        self.mod_align_weight = float(config.get('mod_align_weight', 0.0))
        self.mod_align_use_cosine = bool(config.get('mod_align_use_cosine', True))

        self.embedding_dict = self._init_model()

        # modality projections if features are available
        length = 0
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
            # nn.init.xavier_normal_(self.image_trs.weight)
            length += self.v_feat.shape[1]
        else:
            # self.image_trs = None
            pass
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            # self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
            # nn.init.xavier_normal_(self.text_trs.weight)
            length += self.t_feat.shape[1]
        else:
            # self.text_trs = None
            pass
        # generate intermediate data
        self.all_trs = nn.Linear(length, self.feat_embed_dim)
        nn.init.xavier_normal_(self.all_trs.weight)

        # optional dropout on modality residual branch
        self.mod_dropout = nn.Dropout(p=self.modal_drop_rate) if self.modal_drop_rate > 0.0 else None


        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # precompute an item-degree gate (cold items benefit more from modality)
        if self.use_degree_gate:
            inter = self.interaction_matrix
            item_deg = np.asarray(np.bincount(inter.col, minlength=self.n_items), dtype=np.float32)
            gate = 1.0 / (1.0 + np.log1p(item_deg))  # [0,1]
            self.register_buffer('item_gate', torch.from_numpy(gate).view(-1, 1))

        # parameters initialization
        #self.apply(xavier_uniform_initialization)

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.n_users, self.latent_dim))),
            'item_emb': nn.Parameter(initializer(torch.empty(self.n_items, self.latent_dim)))
        })

        return embedding_dict

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix((self.n_users + self.n_items,
                           self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col+self.n_users),
                             [1]*inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row+self.n_users, inter_M_t.col),
                                  [1]*inter_M_t.nnz)))
        # A._update(data_dict)
        for (row, col), value in data_dict.items():
            A[row, col] = value
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        # add epsilon to avoid Devide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor([row, col])
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_id = self.embedding_dict['user_emb']                         # [U, d]
        item_id = self.embedding_dict['item_emb']                         # [I, d]

        # use learnable modality embeddings if available
        # img = self.image_trs(self.image_embedding.weight) if getattr(self, 'image_trs', None) is not None else None  # [I, d]
        # txt = self.text_trs(self.text_embedding.weight) if getattr(self, 'text_trs', None) is not None else None    # [I, d]

        # Build modality branch (projected concat of available features)
        feats = []
        if hasattr(self, 'image_embedding'):
            feats.append(self.image_embedding.weight)
        if hasattr(self, 'text_embedding'):
            feats.append(self.text_embedding.weight)
        if len(feats) > 0:
            feat_cat = torch.cat(feats, dim=1)
            mod_branch = self.all_trs(feat_cat)
            if self.use_branch_norm:
                mod_branch = F.normalize(mod_branch, dim=1)
            if self.mod_dropout is not None and self.training:
                mod_branch = self.mod_dropout(mod_branch)
            if self.use_degree_gate and hasattr(self, 'item_gate'):
                mod_branch = self.item_gate * mod_branch  # (I,1) * (I,d)
            item_emb = item_id + torch.clamp(self.res_alpha, min=self.res_alpha_min) * mod_branch
        else:
            item_emb = item_id
        ego = torch.cat([user_id, item_emb], dim=0)                       # [(U+I), d]
        return ego

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings = lightgcn_all_embeddings[:self.n_users, :]
        item_all_embeddings = lightgcn_all_embeddings[self.n_users:, :]

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_item = interaction[1]
        neg_item = interaction[2]

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user, :]
        posi_embeddings = item_all_embeddings[pos_item, :]
        negi_embeddings = item_all_embeddings[neg_item, :]

        # calculate BPR Loss
        pos_scores = torch.mul(u_embeddings, posi_embeddings).sum(dim=1)
        neg_scores = torch.mul(u_embeddings, negi_embeddings).sum(dim=1)
        mf_loss = self.mf_loss(pos_scores, neg_scores)

        # calculate BPR Loss
        u_ego_embeddings = self.embedding_dict['user_emb'][user, :]
        posi_ego_embeddings = self.embedding_dict['item_emb'][pos_item, :]
        negi_ego_embeddings = self.embedding_dict['item_emb'][neg_item, :]

        reg_loss = self.reg_loss(u_ego_embeddings, posi_ego_embeddings, negi_ego_embeddings)
        loss = mf_loss + self.reg_weight * reg_loss

        # Optional lightweight alignment loss to prevent modality collapse
        if self.mod_align_weight > 0.0:
            # reconstruct modality branch (same as in get_ego_embeddings)
            feats = []
            if hasattr(self, 'image_embedding'):
                feats.append(self.image_embedding.weight)
            if hasattr(self, 'text_embedding'):
                feats.append(self.text_embedding.weight)
            if len(feats) > 0:
                feat_cat = torch.cat(feats, dim=1)
                mod_branch = self.all_trs(feat_cat)
                if self.use_branch_norm:
                    mod_branch = F.normalize(mod_branch, dim=1)
                if self.use_degree_gate and hasattr(self, 'item_gate'):
                    mod_branch = self.item_gate * mod_branch
                alpha_eff = torch.clamp(self.res_alpha, min=self.res_alpha_min)
                mod_post = alpha_eff * mod_branch
                # pick involved items
                ids = torch.unique(torch.cat([pos_item, neg_item], dim=0))
                mod_sel = mod_post[ids]
                id_sel = self.embedding_dict['item_emb'].detach()[ids]  # stop gradient into ID branch
                if self.mod_align_use_cosine:
                    mod_n = F.normalize(mod_sel, dim=1)
                    id_n = F.normalize(id_sel, dim=1)
                    align_loss = 1.0 - torch.sum(mod_n * id_n, dim=1).mean()
                else:
                    align_loss = F.mse_loss(mod_sel, id_sel)
                loss = loss + self.mod_align_weight * align_loss

        return loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        restore_user_e, restore_item_e = self.forward()
        u_embeddings = restore_user_e[user, :]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, restore_item_e.transpose(0, 1))

        return scores

    @torch.no_grad()
    def tb_diagnostics(self):
        """Lightweight diagnostics to verify modality isn’t silently discarded.

        Logs include:
        - w/res_alpha: current residual weight
        - norm/id: mean L2 norm of ID item embeddings
        - norm/mod_pre: mean L2 norm of projected modality branch before gate/alpha
        - norm/mod_post: mean L2 norm after gate*alpha
        - gate/mean|min|max: gate stats when enabled
        - ratio/mod_vs_id: mean(||alpha*gate*mod|| / (||id||+1e-12))
        - cos/id_mod: mean cosine similarity between ID and modality branch
        """
        diag = {}
        try:
            id_emb = self.embedding_dict['item_emb']  # (I,d)
            id_norm = torch.norm(id_emb, dim=1)
            diag['w/res_alpha'] = float(torch.clamp(self.res_alpha, min=self.res_alpha_min).item())
            diag['norm/id'] = float(id_norm.mean().item())

            # modality branch reconstruction (same as in get_ego_embeddings)
            feats = []
            if hasattr(self, 'image_embedding'):
                feats.append(self.image_embedding.weight)
            if hasattr(self, 'text_embedding'):
                feats.append(self.text_embedding.weight)
            if len(feats) == 0:
                diag['norm/mod_pre'] = 0.0
                diag['norm/mod_post'] = 0.0
                diag['ratio/mod_vs_id'] = 0.0
                diag['cos/id_mod'] = 0.0
                if hasattr(self, 'item_gate'):
                    gate = self.item_gate.view(-1)
                    diag['gate/mean'] = float(gate.mean().item())
                    diag['gate/min'] = float(gate.min().item())
                    diag['gate/max'] = float(gate.max().item())
                return diag

            feat_cat = torch.cat(feats, dim=1)
            mod_branch = self.all_trs(feat_cat)
            if self.use_branch_norm:
                mod_branch = F.normalize(mod_branch, dim=1)
            mod_pre_norm = torch.norm(mod_branch, dim=1)
            diag['norm/mod_pre'] = float(mod_pre_norm.mean().item())

            if self.use_degree_gate and hasattr(self, 'item_gate'):
                mod_branch = self.item_gate * mod_branch
                gate = self.item_gate.view(-1)
                diag['gate/mean'] = float(gate.mean().item())
                diag['gate/min'] = float(gate.min().item())
                diag['gate/max'] = float(gate.max().item())

            alpha = torch.clamp(self.res_alpha, min=0.0)
            mod_post = alpha * mod_branch
            mod_post_norm = torch.norm(mod_post, dim=1)
            diag['norm/mod_post'] = float(mod_post_norm.mean().item())

            ratio = (mod_post_norm / (id_norm + 1e-12)).mean()
            diag['ratio/mod_vs_id'] = float(ratio.item())

            # cosine similarity between id and pre-gate modality direction
            # (use pre-gate to capture semantic alignment; both L2-normalized)
            id_dir = F.normalize(id_emb, dim=1)
            mod_dir = F.normalize(self.all_trs(feat_cat), dim=1)
            cos = torch.sum(id_dir * mod_dir, dim=1)
            diag['cos/id_mod'] = float(cos.mean().item())
        except Exception:
            pass
        return diag
