
Baby dataset multimodal alignment summary (subset N=500)
=======================================================

Cosine verification (AUC):
- CLIP-space: 0.9827
- Raw-space (after linear map): 0.8556

Retrieval (image -> text):
- CLIP-space R@1/R@5/R@10: 0.4900/0.7780/0.8680, Median Rank: 2.0
- Raw-space  R@1/R@5/R@10: 0.1660/0.3120/0.4360, Median Rank: 17.0

Alignment metrics:
- Linear CKA (higher=better): CLIP 0.5206 vs Raw 0.3704
- Raw linear map relative error (lower=better): 0.8325

Figures under images/:
- baby_cosine_hist_rn.png (CLIP-space cosine distributions)
- baby_cosine_hist_raw_linear.png (Raw-space cosine after linear map)
- baby_cosine_stats_bootstrap.png (Bootstrap 95% CI and KS test per space)

Interpretation:
- CLIP特征在未训练的情况下即可进行跨模态相似度计算与检索（高AUC与较高R@K），说明图文被映射到统一语义空间；
- 原始特征维度不一致，需学习线性映射后才可比较；即使如此，验证与检索指标通常仍显著低于CLIP，表明融合难度更大；
- 更高的CKA与更小的线性映射误差（相对）进一步量化了跨模态对齐程度差异。
