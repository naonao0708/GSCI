from __future__ import division
import pandas as pd
from numpy import *
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import cosine

from dl_simulation import *
from analyze_predictions import *

import spams

THREADS = 4
np.random.seed(1)  # 可以选择任何整数作为种子


def compare_results(A, B):
    results = list(correlations(A, B, 0))[:-1]
    results += list(compare_distances(A, B))
    results += list(compare_distances(A.T, B.T))
    return results


# 计算适应度值
def calFitness_DE(X):
    n = len(X)
    fitness = 0
    for i in range(n):
        fitness += X[i] * X[i]
    return fitness


# 用于评估输入和稀疏编码权重之间的相关性
def calFitness(X, UW):
    n = X.shape[1]
    fitness = np.zeros((1, n))
    for i in range(n):
        fitness[0, i] = 1 - pearsonr(X[:, i], UW[:, i])[0]
    return fitness[0]


def calFitness_1(X, UW):
    return 1 - pearsonr(X, UW)[0]


# 常青藤算法 (Ivy Algorithm)
def IvyAlgorithm(sizepop, Ws_tem, fitness, params):
    population = Ws_tem
    for i in range(sizepop):
        # 找到最优邻居的位置
        fmin = np.min(fitness)
        fmin_arg = np.argmin(fitness)
        best_neighbor = population[fmin_arg]

        # 计算 beta 参数
        beta = (2 + np.random.rand()) / 2

        # 更新每个个体的位置
        if fitness[i] > fmin:  # 只有在适应度较差时才更新位置
            population[i] = population[i] + beta * (best_neighbor - population[i]) \
                            + np.random.normal(0, 1, population[i].shape) * (best_neighbor - population[i])
    return population

# 更新选择
def selection(XTemp, XTemp1, fitnessVal, X, U):
    m, n = shape(XTemp)
    fitnessVal1 = zeros(m)
    for i in range(m):
        fitnessVal1[i] = calFitness_1(X, U.dot(XTemp1[i]))
        if (fitnessVal1[i] < fitnessVal[i]):
            for j in range(n):
                XTemp[i, j] = XTemp1[i, j]
            fitnessVal[i] = fitnessVal1[i]
    return XTemp, fitnessVal


def saveBest(fitnessVal, XTemp):
    m = shape(fitnessVal)[0]
    tmp = 0
    for i in range(1, m):
        if (fitnessVal[tmp] > fitnessVal[i]):
            tmp = i
    return fitnessVal[tmp][0], XTemp[tmp]


# 计算 MSE（均方误差）
def calculate_mse(original, reconstructed):
    return mean_squared_error(original, reconstructed)


# 计算 MAE（平均绝对误差）
def calculate_mae(original, reconstructed):
    return mean_absolute_error(original, reconstructed)


# 计算 PCC（皮尔森相关系数）
def calculate_pcc(original, reconstructed):
    pcc, _ = pearsonr(original.flatten(), reconstructed.flatten())
    return pcc


# 计算余弦相似度
def calculate_cs(original, reconstructed):
    cs = 1 - cosine(original.flatten(), reconstructed.flatten())
    return cs
def early_stopping(result_list, patience=1000, threshold=0.001):
    if len(result_list) < patience + 1:
        return False
    last_results = result_list[-patience:]
    first_result = last_results[0]
    last_result = last_results[-1]
    mse_change = abs(first_result['mse'] - last_result['mse']) / first_result['mse']
    mae_change = abs(first_result['mae'] - last_result['mae']) / first_result['mae']
    pcc_change = abs(first_result['pcc'] - last_result['pcc']) / first_result['pcc']
    cs_change = abs(first_result['cs'] - last_result['cs']) / first_result['cs']
    if mse_change < threshold and mae_change < threshold and pcc_change < threshold and cs_change < threshold:
        return True
    return False


# ================== 新增评估函数 ==================
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import issparse
import scanpy as sc
import time
import psutil


def evaluate_zero_recovery(original, reconstructed, threshold=1e-5):
    """评估零值区域恢复质量"""
    zero_mask = (original <= threshold)
    non_zero_mask = ~zero_mask

    metrics = {
        # 零值区域指标
        'zero_ratio_recovered': (np.abs(reconstructed[zero_mask]) < threshold).mean(),
        'zero_mae': mean_absolute_error(original[zero_mask], reconstructed[zero_mask]),
        'zero_mse': mean_squared_error(original[zero_mask], reconstructed[zero_mask]),
        'zero_pcc': pearsonr(original[zero_mask].flatten(), reconstructed[zero_mask].flatten())[0],
        'zero_cs': 1 - cosine(original[zero_mask].flatten(), reconstructed[zero_mask].flatten()),

        # 非零区域指标
        'non_zero_pcc': pearsonr(original[non_zero_mask].flatten(),
                                 reconstructed[non_zero_mask].flatten())[0],
        'non_zero_mae': mean_absolute_error(original[non_zero_mask], reconstructed[non_zero_mask]),
        'non_zero_mse': mean_squared_error(original[non_zero_mask], reconstructed[non_zero_mask]),
        'non_zero_cs': 1 - cosine(original[non_zero_mask].flatten(), reconstructed[non_zero_mask].flatten())
    }
    return metrics


# def biological_consistency(original, reconstructed, n_neighbors=15):
#     """评估细胞聚类一致性（无需真实标签）"""
#     # 创建AnnData对象
#     adata_orig = sc.AnnData(original.T)
#     adata_recon = sc.AnnData(reconstructed.T)
#
#     # 标准化和聚类
#     for adata in [adata_orig, adata_recon]:
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#         sc.pp.pca(adata)
#         sc.pp.neighbors(adata, n_neighbors=n_neighbors)
#         sc.tl.leiden(adata, resolution=0.5)
#
#     # 计算相似性
#     ari = adjusted_rand_score(adata_orig.obs['leiden'], adata_recon.obs['leiden'])
#     return {'clustering_ari': ari}

# def biological_consistency(original, reconstructed):
#     """使用t-SNE替代UMAP进行聚类分析"""
#     import scanpy as sc
#     from sklearn.metrics import adjusted_rand_score
#
#     # 创建AnnData对象并显式指定数据类型
#     adata_orig = sc.AnnData(original.T.astype(np.float32))
#     adata_recon = sc.AnnData(reconstructed.T.astype(np.float32))
#
#     # 统一预处理流程
#     for adata in [adata_orig, adata_recon]:
#         sc.pp.normalize_total(adata, target_sum=1e4)
#         sc.pp.log1p(adata)
#         sc.pp.highly_variable_genes(adata, n_top_genes=2000)
#         sc.pp.pca(adata, n_comps=50)  # 使用PCA初始化
#
#         # 使用t-SNE降维
#         sc.tl.tsne(adata, n_pcs=50, random_state=42)
#
#         # Leiden聚类
#         sc.tl.leiden(adata, resolution=0.5, key_added='cluster')
#
#     # 计算ARI（需要真实标签）
#     # 如果没有真实标签，可删除此行
#     ari = adjusted_rand_score(adata_orig.obs['cluster'], adata_recon.obs['cluster'])
#
#     return {'clustering_ari': ari}

def monitor_resource_usage():
    """监控资源使用"""
    return {
        'time_stamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'memory_usage': psutil.Process().memory_info().rss / 1024 ** 2  # MB
    }


def load_gene_names(gene_file_path):
    """从txt文件加载基因名称列表"""
    with open(gene_file_path, 'r') as f:
        gene_names = [line.strip() for line in f]
    return gene_names

def map_gene_indices(target_genes, all_genes):
    """将基因名称映射为索引"""
    gene_indices = []
    for gene in target_genes:
        try:
            idx = all_genes.index(gene)
            gene_indices.append(idx)
        except ValueError:
            print(f"Warning: Gene {gene} not found in dataset")
    return gene_indices


def evaluate_key_genes_by_name(original, reconstructed, target_genes, all_genes):
    """基于基因名称的评估"""
    # 获取基因索引
    gene_indices = map_gene_indices(target_genes, all_genes)

    # 评估每个基因
    metrics = {}
    for gene, idx in zip(target_genes, gene_indices):
        orig = original[idx, :]
        recon = reconstructed[idx, :]
        metrics[f'{gene}_pcc'] = pearsonr(orig, recon)[0]
        metrics[f'{gene}_mse'] = mean_squared_error(orig, recon)
        metrics[f'{gene}_mae'] = mean_absolute_error(orig, recon)
        metrics[f'{gene}_cs'] = 1 - cosine(orig, recon)
    return metrics


# 在生成xa后的评估部分
def main_evaluation(z, xa, all_genes, target_genes):
    """整合所有评估指标"""
    metrics = {}

    # 基础数值指标
    metrics['mse'] = mean_squared_error(z, xa)
    metrics['mae'] = mean_absolute_error(z, xa)
    metrics['pcc'] = pearsonr(z.flatten(), xa.flatten())[0]
    metrics['cs'] = 1 - cosine(z.flatten(), xa.flatten())

    # 零值恢复专项
    metrics.update(evaluate_zero_recovery(z, xa))

    # # 生物学聚类一致性
    # metrics.update(biological_consistency(z, xa))

    # 关键基因评估
    metrics.update(evaluate_key_genes_by_name(z, xa, target_genes, all_genes))

    # 资源监控
    metrics.update(monitor_resource_usage())

    return metrics


# ================== 在 metrics 计算后添加 ==================
def print_iteration_summary(gen, metrics):
    """格式化打印当前迭代结果"""
    # 基础指标
    print(f"\n=== Iteration {gen + 1} ===")
    print(f"  MSE      : {metrics.get('mse', 0):.4e}")
    print(f"  MAE      : {metrics.get('mae', 0):.4e}")
    print(f"  PCC      : {metrics.get('pcc', 0):.4f}")
    print(f"  CS       : {metrics.get('cs', 0):.4f}")

    # 关键生物指标
    print("  --- Biological Metrics ---")
    print(f"  Clustering ARI : {metrics.get('clustering_ari', 0):.4f}")

    # 关键基因表现（自动筛选所有以_pcc结尾的指标）
    pcc_genes = {k: v for k, v in metrics.items() if k.endswith('_pcc')}
    for gene, val in pcc_genes.items():
        print(f"  {gene.split('_')[0]:<8} : {val:.4f}")

    # 资源使用
    print("  --- Resource Usage ---")
    print(f"  Memory (MB) : {metrics.get('memory_usage', 0):.1f}")
    print(f"  Time (s)    : {time.time() - metrics.get('start_time', 0):.1f}")


# def preprocess_if_vaegan(x, method='sparsify_and_normalize', threshold=0.1):
#     """
#     对VAE-GAN输出进行预处理，使其更适配IVYCS：
#     - method: 'normalize' | 'sparsify' | 'sparsify_and_normalize'
#     """
#     x_processed = x.copy()
#
#     if method == 'normalize':
#         x_min = np.min(x_processed, axis=0)
#         x_max = np.max(x_processed, axis=0)
#         x_processed = (x_processed - x_min) / (x_max - x_min + 1e-8)
#
#     elif method == 'sparsify':
#         x_processed[np.abs(x_processed) < threshold] = 0
#
#     elif method == 'sparsify_and_normalize':
#         # 先稀疏化再归一化
#         x_processed[np.abs(x_processed) < threshold] = 0
#         x_min = np.min(x_processed, axis=0)
#         x_max = np.max(x_processed, axis=0)
#         x_processed = (x_processed - x_min) / (x_max - x_min + 1e-8)
#
#     return x_processed

import numpy as np
from scipy.ndimage import gaussian_filter


def soft_thresholding(x, threshold=0.1):
    """软阈值稀疏化"""
    return np.sign(x) * np.maximum(np.abs(x) - threshold, 0)


def normalize_columns(x):
    """按列归一化到 [0,1]"""
    x_min = np.min(x, axis=0)
    x_max = np.max(x, axis=0)
    return (x - x_min) / (x_max - x_min + 1e-8)

def merge_vaegan_ivycs_weighted(x1, x3, original_x, ground_truth, dropout_threshold=1e-5, alpha=5.0):
    """
    融合 VAE-GAN 和 IVYCS 输出（soft-weighted）：
    - 所有 Dropout 区域做加权平均，权重根据 VAE-GAN 恢复误差动态调整

    参数：
        x1: VAE-GAN 输出
        x3: IVYCS 输出
        original_x: 原始输入（含 Dropout）
        ground_truth: 参考真值（用于误差计算）
        dropout_threshold: 判断 Dropout 的阈值
        alpha: 控制权重敏感度

    返回：
        融合后的结果矩阵
    """
    assert x1.shape == x3.shape == original_x.shape == ground_truth.shape

    # Dropout 掩码
    dropout_mask = original_x < dropout_threshold

    # 初始化结果
    merged = x1.copy()

    # 在 Dropout 区域做误差加权融合
    diff1 = np.abs(x1 - ground_truth)
    diff2 = np.abs(x3 - ground_truth)

    # 计算动态权重（小误差 → 大权重）
    eps = 1e-8
    w1 = np.exp(-alpha * diff1)
    w2 = np.exp(-alpha * diff2)
    w_sum = w1 + w2

    w1 = w1 / (w_sum + eps)
    w2 = w2 / (w_sum + eps)

    # 只在 dropout 区域做加权融合
    merged[dropout_mask] = w1[dropout_mask] * x1[dropout_mask] + w2[dropout_mask] * x3[dropout_mask]

    return merged


from run_smaf import smaf
if __name__ == "__main__":
    # SMAF setting
    biased_training = 0.  # 偏置
    composition_noise = 0.  # 噪声
    subset_size = 0  # 子集大小
    measurements = 400  # 测量的数量
    sparsity = 10  # 稀疏度
    dictionary_size = 0.5  # 字典大小
    training_dictionary_fraction = 0.05  # 训练字典比例
    SNR = 2.0  # 信噪比
    ####################增加##############################
    best_mse = float('inf')  # 初始化最佳MSE
    best_mae = float('inf')  # 初始化最佳MAE
    best_pcc = float('-inf')  # 初始化最佳PCC
    best_cs = float('-inf')  # 初始化最佳CS
    early_stop_counter = 0  # 用于记录早停的迭代次数
    early_stop_patience = 1000  # 设定耐心值，超过该次数没有改进就停止
    result1 = []  # 用于保存每轮的结果
    result2 = []  # 用于保存每轮的结果

    # 数据加载
    # x = np.load("./Result/GSE147326/results/GSE147326_VAE-GAN.npy")
    x = np.load("./Result/GSE124989/results/GSE124989_VAE-GAN.npy")
    # x = np.load("./Result/GSE123358/results/GSE123358_VAE-GAN.npy")
    # x = np.load('../Dropout_Study/Result/GSE124989/GSE124989_VAE-GAN_drop10.npy')
    x=x.T

    x = x.astype(np.float64)
    # z = np.load("./Data/GSE147326/GSE147326_log.npy")
    z = np.load("./Data/GSE124989/GSE124989_log.npy")
    # z = np.load("./Data/GSE123358/GSE123358_log.npy")
    z = z.astype(np.float64)

    print(f"原始数据形状: {x.shape}")
    print(f"原始数据形状: {z.shape}" )

    # 加载基因名称
    # all_genes = load_gene_names("./Data/GSE147326/gene.txt")  # 原始数据基因名称
    # all_genes = load_gene_names("./Data/GSE123358/gene.txt")  # 原始数据基因名称
    all_genes = load_gene_names("./Data/GSE124989/gene.txt")  # 原始数据基因名称
    target_genes = ['BRCA1', 'BRCA2', 'BARD1', 'BRIP1', 'PALB2', 'RAD51', 'RAD54L', 'XRCC3', 'ERBB2', 'ESR1', 'PGR',
                        'PIK3CA', 'TP53', 'PPM1D', 'RB1CC1', 'HMMR', 'NQO2', 'SLC22A18', 'PTEN', 'EGFR', 'KIT', 'NOTCH1',
                        'FZD7', 'LRP6', 'FGFR1', 'CCND1']
    print(f"关键基因匹配情况: {len(map_gene_indices(target_genes, all_genes))}/{len(target_genes)}")

    itr = 0
    while (itr < 1):
        # 参数设置
        NP = 40  # 种群大小
        maxItr = 100

        # 初始化
        k = min(int(x.shape[1] * 3), 20)  # 稀疏编码的基函数数量
        Ws = np.zeros((NP, k, x.shape[1]))
        UW = (np.random.random((x.shape[0], k)), np.random.random((k, x.shape[1])))
        # UF, WF = smaf(x, k, 5, 0.0001, maxItr=10, use_chol=True, activity_lower=0., module_lower=x.shape[0] / 10, UW=UW,
        #               donorm=True, mode=1, mink=3.)
        UF, WF = smaf(x, k, 5, 0.0001, maxItr=10, use_chol=True, activity_lower=0., module_lower=x.shape[0] / 10, UW=UW,
                      donorm=True, mode=1, mink=3.)

        # 生成初始种群
        for i in range(NP):
            lda = np.random.randint(5, 20)
            Ws[i] = sparse_decode(x, UF, lda, worstFit=1 - 0.0005, mink=3.)

        # 计算适应度值
        fitnessVal = zeros((NP, x.shape[1]))
        for i in range(NP):
            fitnessVal[i] = calFitness(x, UF.dot(Ws[i]))

        gen = 0
        Xnorm = np.linalg.norm(x) ** 2 / x.shape[1]
        while gen <= maxItr:
            for i in range(x.shape[1]):
                Ws_tem = Ws[:, :, i]
                fmin = np.min(fitnessVal[:, i])
                fmin_arg = np.argmin(fitnessVal[:, i])
                best = Ws_tem[fmin_arg, :]

                # 使用常青藤算法优化
                fa = IvyAlgorithm(NP, Ws_tem, fitnessVal[:, i], [1.0, 1.0])
                Ws_tem, fitnessVal[:, i] = selection(Ws_tem, fa, fitnessVal[:, i], x[:, i], UF)

                WF[:, i] = Ws_tem[np.where(fitnessVal[:, i] == min(fitnessVal[:, i]))[0][0], :]
                Ws[:, :, i] = Ws_tem

            UF = spams.lasso(np.asfortranarray(x.T), D=np.asfortranarray(WF.T),
                             lambda1=0.0005 * Xnorm, mode=1, numThreads=THREADS, cholesky=True, pos=True)
            UF = np.asarray(UF.todense()).T
            # print(gen)
            xa, phi, y, w, d, psi = recover_system_knownBasis(x, measurements, sparsity, Psi=UF, snr=SNR,
                                                              use_ridge=False)
            xa = merge_vaegan_ivycs_weighted(
                x1=x,
                x3=xa,
                original_x=z,
                ground_truth=z,
                dropout_threshold=1e-5,
                alpha=8.0
            )

            # 执行完整评估
            start_time = time.time()
            metrics = main_evaluation(z, xa, all_genes, target_genes)
            metrics['start_time'] = start_time  # 记录评估开始时间

            # 打印当前迭代结果
            print_iteration_summary(gen, metrics)
            result1.append({'iteration': gen + 1,**metrics})
            # 输出当前轮次的结果
            print(f"Iteration {gen+1}: metrics={metrics}")
            if early_stopping(result1, patience=early_stop_patience):
                break
            gen += 1

        # 结果比较
        Results = {}
        xa, phi, y, w, d, psi = recover_system_knownBasis(x, measurements, sparsity, Psi=UF, snr=SNR, use_ridge=False)
        Results['Ivy'] = compare_results(x, xa)
        for k, v in sorted(Results.items()):
            print('\t'.join([k] + [str(x) for x in v]))

        itr += 1

        # np.savetxt('./Result/GSE147326/results/GSE147326_VAE+GAN+IVY.npy', xa, delimiter=',')
        np.savetxt('./Result/GSE124989/results/GSE124989_VAE+GAN+IVY.npy', xa, delimiter=',')

        df = pd.DataFrame(xa)
        df.to_csv("./Result/GSE124989/GSE124989_VAE+GAN+IVY.csv")
        np.save('./Result/GSE124989/GSE124989_VAE+GAN+IVY.npy', xa)
        # # np.savetxt('./Result/GSE124989/GSE124989_IVY.csv', xa, delimiter=',')
        # np.savetxt('./Result/GSE123358/results/GSE123358_VAE-GAN-IVY.npy', xa, delimiter=',')
        # np.savetxt('./Result/GSE123358/results/GSE123358_VAE-GAN-IVY.csv', xa, delimiter=',')
        df = pd.DataFrame(result1)

        # 取平均
        # 添加平均值（自动取所有迭代轮次）
        avg_row = df[df['iteration'] != 'avg_all'].mean(numeric_only=True)
        avg_row['iteration'] = 'avg_all'
        df = pd.concat([df, pd.DataFrame([avg_row])], ignore_index=True)

        # 保存全局指标（iteration/mse/mae/pcc/cs 等）
        global_metrics = ['iteration', 'mse', 'mae', 'pcc', 'cs', 'zero_mae', 'zero_mse', 'zero_pcc', 'zero_cs',
                          'non_zero_mae', 'non_zero_mse', 'non_zero_pcc', 'non_zero_cs']
        df_global = df[[col for col in df.columns if col in global_metrics]]
        # df_global.to_csv("./Result/GSE147326/results/GSE147326_VAE+GAN+IVY_global.csv", index=False)
        df_global.to_csv("./Result/GSE124989/GSE124989_VAE+GAN+IVY_global.csv", index=False)
        # df_global.to_csv("./Result/GSE123358/results/GSE123358_VAE+GAN+IVY_global.csv", index=False)

        gene_metrics = [
            col for col in df.columns
            if any(col.endswith(suffix) for suffix in ['_mae', '_mse', '_pcc', '_cs'])
               and not col.startswith(('zero', 'non_zero', 'memory', 'start', 'usage', 'stamp', 'time'))
        ]

        # 基因指标行（只取最后平均那一行）
        gene_data = df.loc[df['iteration'] == 'avg_all', gene_metrics].T.reset_index()
        gene_data.columns = ['metric_name', 'value']

        # 提取基因名和指标名
        gene_data['gene'] = gene_data['metric_name'].apply(lambda x: x.split('_')[0])
        gene_data['metric'] = gene_data['metric_name'].apply(lambda x: '_'.join(x.split('_')[1:]))

        # ✅ 仅保留在目标基因列表中的
        gene_data = gene_data[gene_data['gene'].isin(target_genes)]

        # Pivot 成格式：行 = gene，列 = 指标
        df_gene = gene_data.pivot(index='gene', columns='metric', values='value').reset_index()
        # df_gene.to_csv("./Result/GSE147326/results/GSE147326_VAE+GAN+IVY_gene.csv", index=False)
        df_gene.to_csv("./Result/GSE124989/GSE124989_VAE+GAN+IVY_gene.csv", index=False)
        # df_gene.to_csv("./Result/GSE123358/results/GSE123358_VAE+GAN+IVY_gene.csv", index=False)

