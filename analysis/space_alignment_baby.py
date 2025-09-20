import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / np.maximum(n, eps)


def sample_negatives(n: int, total: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    i = rng.randint(0, total, size=n)
    j = rng.randint(0, total, size=n)
    mask = i == j
    while mask.any():
        j[mask] = rng.randint(0, total, size=mask.sum())
        mask = i == j
    return i, j


def auc_from_scores(pos: np.ndarray, neg: np.ndarray) -> float:
    # Mann–Whitney U equivalent AUC with proper tie handling
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)])
    order = np.argsort(x)
    x_sorted = x[order]
    ranks_sorted = np.arange(1, len(x_sorted) + 1, dtype=float)
    # average ties on ranks_sorted
    start = 0
    while start < len(x_sorted):
        end = start + 1
        while end < len(x_sorted) and x_sorted[end] == x_sorted[start]:
            end += 1
        avg = 0.5 * (start + 1 + end)
        ranks_sorted[start:end] = avg
        start = end
    ranks = np.empty_like(ranks_sorted)
    ranks[order] = ranks_sorted
    r_pos = ranks[y == 1].sum()
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    auc = (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def retrieval_metrics(xn: np.ndarray, yn: np.ndarray, ks=(1, 5, 10)) -> tuple[dict, float]:
    n = xn.shape[0]
    yt = yn.T.astype(np.float32, copy=False)
    hits = {k: 0 for k in ks}
    ranks = np.empty(n, dtype=np.int32)
    bsz = 128
    row = 0
    while row < n:
        end = min(row + bsz, n)
        sim = xn[row:end] @ yt  # [B, N]
        # top-k
        for k in ks:
            topk = np.argpartition(-sim, kth=min(k, sim.shape[1]-1), axis=1)[:, :k]
            idx = np.arange(row, end)
            hits[k] += int((topk == idx[:, None]).any(axis=1).sum())
        # exact rank for median rank
        order = np.argsort(-sim, axis=1)
        for i, ord_row in enumerate(order):
            ranks[row + i] = int(np.where(ord_row == (row + i))[0][0]) + 1
        row = end
    rec = {k: hits[k] / float(n) for k in ks}
    medr = float(np.median(ranks))
    return rec, medr


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x_c = x - x.mean(axis=0, keepdims=True)
    y_c = y - y.mean(axis=0, keepdims=True)
    xTy = x_c.T @ y_c
    xTx = x_c.T @ x_c
    yTy = y_c.T @ y_c
    num = np.linalg.norm(xTy, 'fro') ** 2
    den = np.linalg.norm(xTx, 'fro') * np.linalg.norm(yTy, 'fro')
    return float(num / den) if den != 0 else 0.0


def fit_ridge_linear_map(x: np.ndarray, y: np.ndarray, lam: float = 1e-2) -> tuple[np.ndarray, float]:
    # Map x -> y
    x_c = x - x.mean(axis=0, keepdims=True)
    y_c = y - y.mean(axis=0, keepdims=True)
    xtx = x_c.T @ x_c
    dx = xtx.shape[0]
    xtx_reg = xtx + lam * np.eye(dx, dtype=x.dtype)
    xty = x_c.T @ y_c
    w = np.linalg.solve(xtx_reg, xty)
    rel_err = np.linalg.norm(x_c @ w - y_c, 'fro') / max(1e-9, np.linalg.norm(y_c, 'fro'))
    return w, float(rel_err)


def plot_hist(pos: np.ndarray, neg: np.ndarray, out_path: str, title: str):
    plt.figure(figsize=(7, 5), dpi=150)
    plt.hist(neg, bins=50, alpha=0.6, label='mismatched', color='#d62728')
    plt.hist(pos, bins=50, alpha=0.6, label='matched', color='#2ca02c')
    plt.legend(frameon=False)
    plt.xlabel('cosine similarity')
    plt.ylabel('count')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def bootstrap_ci_1d(x: np.ndarray, stat_fn, n_boot: int = 1000, rng: np.random.RandomState | None = None,
                    ci: float = 0.95) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.RandomState(123)
    n = len(x)
    point = float(stat_fn(x))
    bs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.randint(0, n, size=n)
        bs[b] = float(stat_fn(x[idx]))
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(bs, alpha)), float(np.quantile(bs, 1 - alpha))
    return point, lo, hi


def bootstrap_auc_ci(pos: np.ndarray, neg: np.ndarray, n_boot: int = 1000,
                     rng: np.random.RandomState | None = None, ci: float = 0.95) -> tuple[float, float, float]:
    if rng is None:
        rng = np.random.RandomState(123)
    n_pos, n_neg = len(pos), len(neg)
    point = auc_from_scores(pos, neg)
    bs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        ip = rng.randint(0, n_pos, size=n_pos)
        ineg = rng.randint(0, n_neg, size=n_neg)
        bs[b] = auc_from_scores(pos[ip], neg[ineg])
    alpha = (1.0 - ci) / 2.0
    lo, hi = float(np.quantile(bs, alpha)), float(np.quantile(bs, 1 - alpha))
    return float(point), lo, hi


def ks_2samp(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    # Two-sample KS test (asymptotic p-value approximation)
    x = np.sort(np.asarray(x, dtype=np.float64))
    y = np.sort(np.asarray(y, dtype=np.float64))
    n, m = len(x), len(y)
    i = j = 0
    cdf_x = cdf_y = 0.0
    d = 0.0
    while i < n and j < m:
        if x[i] <= y[j]:
            i += 1
            cdf_x = i / n
        else:
            j += 1
            cdf_y = j / m
        d = max(d, abs(cdf_x - cdf_y))
    # Drain tails
    while i < n:
        i += 1
        cdf_x = i / n
        d = max(d, abs(cdf_x - cdf_y))
    while j < m:
        j += 1
        cdf_y = j / m
        d = max(d, abs(cdf_x - cdf_y))
    # Asymptotic p-value
    en = np.sqrt(n * m / (n + m))
    lam = (en + 0.12 + 0.11 / en) * d
    # Kolmogorov distribution tail (one minus CDF)
    # Q_KS(lam) = 2 * sum_{k=1..inf} (-1)^{k-1} exp(-2 k^2 lam^2)
    # truncate when terms negligible
    s = 0.0
    k = 1
    while True:
        term = 2.0 * ((-1) ** (k - 1)) * np.exp(-2.0 * (k * k) * (lam * lam))
        s += term
        if abs(term) < 1e-8 or k > 1000:
            break
        k += 1
    p = float(min(max(s, 0.0), 1.0))
    return float(d), p


def plot_with_stats(pos: np.ndarray, neg: np.ndarray, title: str, ax: plt.Axes,
                    rng: np.random.RandomState, n_boot: int = 1000):
    ax.hist(neg, bins=50, alpha=0.6, label='mismatched', color='#d62728')
    ax.hist(pos, bins=50, alpha=0.6, label='matched', color='#2ca02c')
    ax.set_title(title)
    ax.set_xlabel('cosine similarity')
    ax.set_ylabel('count')
    # stats
    mean_pos, lo_mpos, hi_mpos = bootstrap_ci_1d(pos, np.mean, n_boot, rng)
    var_pos, lo_vpos, hi_vpos = bootstrap_ci_1d(pos, lambda z: np.var(z, ddof=1), n_boot, rng)
    mean_neg, lo_mneg, hi_mneg = bootstrap_ci_1d(neg, np.mean, n_boot, rng)
    var_neg, lo_vneg, hi_vneg = bootstrap_ci_1d(neg, lambda z: np.var(z, ddof=1), n_boot, rng)
    auc_pt, auc_lo, auc_hi = bootstrap_auc_ci(pos, neg, n_boot, rng)
    d_stat, p_val = ks_2samp(pos, neg)
    text = (
        f"matched mean: {mean_pos:.3f} [{lo_mpos:.3f}, {hi_mpos:.3f}]\n"
        f"matched var:  {var_pos:.3f} [{lo_vpos:.3f}, {hi_vpos:.3f}]\n"
        f"mismatch mean:{mean_neg:.3f} [{lo_mneg:.3f}, {hi_mneg:.3f}]\n"
        f"mismatch var: {var_neg:.3f} [{lo_vneg:.3f}, {hi_vneg:.3f}]\n"
        f"AUC:          {auc_pt:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]\n"
        f"KS D={d_stat:.3f}, p={p_val:.3g}"
    )
    ax.legend(frameon=False)
    ax.text(0.98, 0.98, text, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, family='monospace', bbox=dict(boxstyle='round', fc='white', ec='0.8', alpha=0.9))


def main():
    base = os.path.join('data', 'baby')
    out_fig = os.path.join('images')
    out_res = os.path.join('evaluation')
    os.makedirs(out_fig, exist_ok=True)
    os.makedirs(out_res, exist_ok=True)

    # Load as float32
    img_raw = np.load(os.path.join(base, 'image_feat_raw.npy')).astype(np.float32, copy=False)
    txt_raw = np.load(os.path.join(base, 'text_feat_raw.npy')).astype(np.float32, copy=False)
    img_rn = np.load(os.path.join(base, 'image_feat_rn.npy')).astype(np.float32, copy=False)
    txt_rn = np.load(os.path.join(base, 'text_feat_rn.npy')).astype(np.float32, copy=False)

    n_all = img_rn.shape[0]
    assert txt_rn.shape[0] == n_all and img_raw.shape[0] == n_all and txt_raw.shape[0] == n_all

    # Subsample for compute efficiency (eval split)
    rng = np.random.RandomState(123)
    eval_n = min(500, n_all)
    eval_idx = rng.choice(n_all, size=eval_n, replace=False)
    remaining = np.setdiff1d(np.arange(n_all), eval_idx, assume_unique=False)
    train_n = min(2000, max(0, len(remaining)))
    if train_n > 0:
        train_idx = rng.choice(remaining, size=train_n, replace=False)
    else:
        # fallback: split eval into two halves
        half = eval_n // 2
        train_idx = eval_idx[:half]
        eval_idx = eval_idx[half:]
        eval_n = len(eval_idx)

    # Slice eval views
    img_rn = img_rn[eval_idx]
    txt_rn = txt_rn[eval_idx]
    img_raw_eval = img_raw[eval_idx]
    txt_raw_eval = txt_raw[eval_idx]
    n = eval_n

    # Normalize
    img_rn_n = l2_normalize(img_rn)
    txt_rn_n = l2_normalize(txt_rn)

    # Matched and mismatched cosine for CLIP-space
    pos_rn = np.sum(img_rn_n * txt_rn_n, axis=1)
    neg_i, neg_j = sample_negatives(n=min(5000, n * 10), total=n, rng=rng)
    # compute neg in chunks to save memory
    def compute_neg(xn, yn):
        out = np.empty(len(neg_i), dtype=np.float32)
        bs = 1000
        for s in range(0, len(neg_i), bs):
            e = min(s + bs, len(neg_i))
            xi = xn[neg_i[s:e]]
            yj = yn[neg_j[s:e]]
            out[s:e] = np.sum(xi * yj, axis=1)
        return out

    neg_rn = compute_neg(img_rn_n, txt_rn_n)
    auc_rn = auc_from_scores(pos_rn, neg_rn)

    # Retrieval for CLIP-space
    rec_rn, medr_rn = retrieval_metrics(img_rn_n, txt_rn_n)

    # Raw: learn a tiny linear map text->image on train set; evaluate on eval set
    w_raw, lin_err_raw = fit_ridge_linear_map(txt_raw[train_idx], img_raw[train_idx], lam=1e-2)
    txt_raw_aligned = (txt_raw_eval - txt_raw[train_idx].mean(axis=0, keepdims=True)) @ w_raw + img_raw[train_idx].mean(axis=0, keepdims=True)
    img_raw_n = l2_normalize(img_raw_eval)
    txt_raw_aligned_n = l2_normalize(txt_raw_aligned)

    pos_raw = np.sum(img_raw_n * txt_raw_aligned_n, axis=1)
    neg_raw = compute_neg(img_raw_n, txt_raw_aligned_n)
    auc_raw = auc_from_scores(pos_raw, neg_raw)

    rec_raw, medr_raw = retrieval_metrics(img_raw_n, txt_raw_aligned_n)

    # CKA alignment
    cka_rn = linear_cka(img_rn, txt_rn)
    cka_raw = linear_cka(img_raw_eval, txt_raw_eval)

    # Plots
    plot_hist(pos_rn, neg_rn, os.path.join(out_fig, 'baby_cosine_hist_rn.png'), 'CLIP-space cosine: matched vs mismatched')
    plot_hist(pos_raw, neg_raw, os.path.join(out_fig, 'baby_cosine_hist_raw_linear.png'), 'Raw-space (after linear map) cosine: matched vs mismatched')

    # Stability figure with bootstrap CIs and KS test
    rng2 = np.random.RandomState(2024)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150, sharey=True)
    plot_with_stats(pos_rn, neg_rn, 'CLIP-space (bootstrap 95% CI + KS)', axes[0], rng2, n_boot=1000)
    plot_with_stats(pos_raw, neg_raw, 'Raw-space linear (bootstrap 95% CI + KS)', axes[1], rng2, n_boot=1000)
    plt.tight_layout()
    stats_fig_path = os.path.join(out_fig, 'baby_cosine_stats_bootstrap.png')
    plt.savefig(stats_fig_path)
    plt.close(fig)

    report = f"""
Baby dataset multimodal alignment summary (subset N={n})
=======================================================

Cosine verification (AUC):
- CLIP-space: {auc_rn:.4f}
- Raw-space (after linear map): {auc_raw:.4f}

Retrieval (image -> text):
- CLIP-space R@1/R@5/R@10: {rec_rn[1]:.4f}/{rec_rn[5]:.4f}/{rec_rn[10]:.4f}, Median Rank: {medr_rn:.1f}
- Raw-space  R@1/R@5/R@10: {rec_raw[1]:.4f}/{rec_raw[5]:.4f}/{rec_raw[10]:.4f}, Median Rank: {medr_raw:.1f}

Alignment metrics:
- Linear CKA (higher=better): CLIP {cka_rn:.4f} vs Raw {cka_raw:.4f}
- Raw linear map relative error (lower=better): {lin_err_raw:.4f}

Figures under images/:
- baby_cosine_hist_rn.png (CLIP-space cosine distributions)
- baby_cosine_hist_raw_linear.png (Raw-space cosine after linear map)
- baby_cosine_stats_bootstrap.png (Bootstrap 95% CI and KS test per space)

Interpretation:
- CLIP特征在未训练的情况下即可进行跨模态相似度计算与检索（高AUC与较高R@K），说明图文被映射到统一语义空间；
- 原始特征维度不一致，需学习线性映射后才可比较；即使如此，验证与检索指标通常仍显著低于CLIP，表明融合难度更大；
- 更高的CKA与更小的线性映射误差（相对）进一步量化了跨模态对齐程度差异。
"""

    with open(os.path.join(out_res, 'baby_alignment_report.md'), 'w') as f:
        f.write(report)

    print(report)


if __name__ == '__main__':
    main()
