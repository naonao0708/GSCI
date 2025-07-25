from __future__ import division
import pandas as pd
from matplotlib import pyplot as plt
from numpy import *
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

    # ✅ 添加字段存在性检查
    required_keys = ['mse', 'mae', 'pcc', 'cs']
    if not all(k in first_result and k in last_result for k in required_keys):
        return False

    # ✅ 处理除零错误
    mse_change = abs(first_result['mse'] - last_result['mse']) / (first_result['mse'] + 1e-8)
    mae_change = abs(first_result['mae'] - last_result['mae']) / (first_result['mae'] + 1e-8)
    pcc_change = abs(first_result['pcc'] - last_result['pcc']) / (abs(first_result['pcc']) + 1e-8)
    cs_change = abs(first_result['cs'] - last_result['cs']) / (abs(first_result['cs']) + 1e-8)

    return all(change < threshold for change in [mse_change, mae_change, pcc_change, cs_change])


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


# 在evaluate_key_genes_by_name中添加调试输出
def evaluate_key_genes_by_name(original, reconstructed, target_genes, all_genes):
    metrics = {}
    gene_indices = map_gene_indices(target_genes, all_genes)

    for gene, idx in zip(target_genes, gene_indices):
        orig = original[idx, :].astype(np.float64)
        recon = reconstructed[idx, :].astype(np.float64)

        # 调试输出数据统计
        print(f"\n=== {gene} (idx={idx}) ===")
        print(f"原始数据: 均值={orig.mean():.2e} 方差={orig.var():.2e} 范围=[{orig.min():.2e}, {orig.max():.2e}]")
        print(f"重构数据: 均值={recon.mean():.2e} 方差={recon.var():.2e} 范围=[{recon.min():.2e}, {recon.max():.2e}]")

        # 检查方差是否为0
        if orig.var() < 1e-6 or recon.var() < 1e-6:
            print(f"⚠️ 方差过低，无法计算PCC")
            metrics[f'{gene}_pcc'] = 0.0
            continue

        # 计算PCC
        pcc = pearsonr(orig, recon)[0]
        metrics[f'{gene}_pcc'] = pcc if not np.isnan(pcc) else 0.0

    return metrics

# 修改后：添加异常处理
def calculate_pcc(original, reconstructed):
    try:
        pcc, _ = pearsonr(original.flatten(), reconstructed.flatten())
        return pcc if not np.isnan(pcc) else 0.0
    except:
        return 0.0
def monitor_resource_usage(start_time):
    return {
        'time_elapsed': time.time() - start_time,
        'memory_usage': psutil.Process().memory_info().rss / 1024**2
    }
def plot_gene_distribution(original, reconstructed, gene_idx):
    plt.figure(figsize=(10, 5))
    plt.scatter(original[gene_idx, :], reconstructed[gene_idx, :], alpha=0.3)
    plt.xlabel("Original Expression")
    plt.ylabel("Reconstructed Expression")
    plt.title(f"Gene {gene_idx} Distribution")
    plt.show()
# 在生成xa后的评估部分
def main_evaluation(z, xa, all_genes, target_genes):
    start_time = time.time()
    """整合所有评估指标"""
    metrics = {}
    # 打印前5个目标基因的索引和名称
    gene_indices = map_gene_indices(target_genes, all_genes)
    print("关键基因索引示例：")
    for gene, idx in zip(target_genes[:5], gene_indices[:5]):
        print(f"{gene}: 索引={idx}")

    # 基础数值指标
    metrics['mse'] = mean_squared_error(z, xa)
    metrics['mae'] = mean_absolute_error(z, xa)
    metrics['pcc'] = calculate_pcc(z, xa)
    # metrics['pcc'] = pearsonr(z.flatten(), xa.flatten())[0]
    metrics['cs'] = 1 - cosine(z.flatten(), xa.flatten())

    # 零值恢复专项
    metrics.update(evaluate_zero_recovery(z, xa))

    # # 生物学聚类一致性
    # metrics.update(biological_consistency(z, xa))

    # 关键基因评估
    metrics.update(evaluate_key_genes_by_name(z, xa, target_genes, all_genes))

    # 资源监控
    metrics.update(monitor_resource_usage(start_time))
    # 在main_evaluation中添加数据统计
    print(f"原始数据非零值比例: {np.mean(z > 1e-5):.2%}")
    print(f"重构数据非零值比例: {np.mean(xa > 1e-5):.2%}")
    # 在评估函数中调用
    plot_gene_distribution(z, xa, 16913)  # BRCA1的索引
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


# ================== 新增降维模块 ==================
from sklearn.decomposition import PCA, FastICA, NMF, TruncatedSVD, FactorAnalysis, KernelPCA, SparsePCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy import sparse

# ================== 新增工具函数 ==================
def fortranify(X):
    """✅ 修改1：强制转换为Fortran格式 + float64"""
    if sparse.issparse(X):
        return X.astype(np.float64).tocsc()
    else:
        return np.asfortranarray(X.astype(np.float64))
class LinearDimReductor:
    # """支持8种线性降维方法的统一接口"""
    #
    # def __init__(self, method='pca', n_components=20):
    #     self.method = method.lower()
    #     self.n_components = n_components
    #     self.model = None
    #     self.shape_ = None  # 记录原始数据形状
    #
    # def _validate_data(self, X):
    #     """统一数据格式为 (features, samples)"""
    #     if X.shape[0] < X.shape[1]:
    #         return X.T
    #     return X
    #
    # def fit_transform(self, X):
    #     """执行降维，返回 (n_components, samples)"""
    #     X = self._validate_data(X)
    #     self.shape_ = X.shape
    #
    #     try:
    #         if self.method == 'pca':
    #             self.model = PCA(n_components=self.n_components)
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         elif self.method == 'ica':
    #             self.model = FastICA(n_components=self.n_components, random_state=0)
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         elif self.method == 'nmf':
    #             # 处理负值问题
    #             X_shifted = X - np.min(X) + 1e-6
    #             self.model = NMF(n_components=self.n_components,
    #                              init='nndsvd',
    #                              max_iter=1000,
    #                              beta_loss='kullback-leibler')
    #             W = self.model.fit_transform(X_shifted)
    #             reduced = self.model.components_
    #             return fortranify(reduced)
    #
    #         elif self.method == 'svd':
    #             self.model = TruncatedSVD(n_components=self.n_components)
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         elif self.method == 'factor':
    #             self.model = FactorAnalysis(n_components=self.n_components)
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         elif self.method == 'kernel_pca':
    #             self.model = KernelPCA(n_components=self.n_components, kernel='rbf')
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         elif self.method == 'lda':
    #             # 需要伪标签，使用聚类生成
    #             from sklearn.cluster import KMeans
    #             pseudo_labels = KMeans(n_clusters=2).fit_predict(X.T)
    #             self.model = LinearDiscriminantAnalysis(n_components=self.n_components)
    #             reduced = self.model.fit_transform(X.T, pseudo_labels).T
    #
    #         elif self.method == 'sparse_pca':
    #             self.model = SparsePCA(n_components=self.n_components, alpha=0.5)
    #             reduced = self.model.fit_transform(X.T).T
    #
    #         else:
    #             raise ValueError(f"不支持的降维方法: {self.method}")
    #
    #         return reduced.astype(np.float32)
    #
    #     except Exception as e:
    #         print(f"{self.method} 降维失败: {str(e)}")
    #         return X  # 返回原始数据作为fallback
    #
    # def inverse_transform(self, reduced):
    #     """逆变换回原始特征空间"""
    #     if self.model is None:
    #         return reduced
    #
    #     try:
    #         if self.method in ['pca', 'ica', 'svd', 'factor', 'lda','sparse_pca','kernel_pca','nmf']:
    #             return self.model.inverse_transform(reduced.T).T
    #
    #         elif self.method == 'nmf':
    #             return self.model.components_.T @ reduced
    #
    #         elif self.method == 'kernel_pca':
    #             print("警告: KernelPCA无精确逆变换，使用近似重构")
    #             from sklearn.linear_model import LinearRegression
    #             reg = LinearRegression().fit(self.model.transform(X.T), X.T)
    #             return reg.predict(reduced.T).T
    #
    #         elif self.method == 'sparse_pca':
    #             return self.model.inverse_transform(reduced.T).T
    #
    #         else:
    #             return reduced
    #
    #     except AttributeError:
    #         return reduced
    """✅ 修改2：简化降维类，确保输入输出维度一致"""

    def __init__(self, method='pca', n_components=20):
        self.method = method.lower()
        self.n_components = n_components
        self.model = None

    def fit_transform(self, X):
        """输入输出形状：(features, samples) -> (components, samples)"""
        try:
            if self.method == 'pca':
                self.model = PCA(n_components=self.n_components)
                return self.model.fit_transform(X.T).T.astype(np.float64)
            elif self.method == 'nmf':
                self.model = NMF(n_components=self.n_components, init='nndsvd')
                return self.model.fit_transform(X.T).T.astype(np.float64)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
        except Exception as e:
            print(f"{self.method}降维失败: {str(e)}")
            return X.astype(np.float64)

    def inverse_transform(self, reduced):
        if self.method == 'pca':
            # ✅ 正确的逆变换（移除错误稀疏化）
            return self.model.inverse_transform(reduced.T).T.astype(np.float64)
        elif self.method == 'nmf':
            return (self.model.components_ @ reduced).astype(np.float64)
        return reduced

# ================== 新增功能：资源监控 ==================
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
    # x = np.load("./Data/GSE123358/GSE123358_log.npy")
    # x = np.load("./Data/GSE124989/GSE124989_log.npy")
    x = np.load("./Data/GSE123358/GSE123358_VAE+GAN1.npy")
    # x = np.load("./Data/GSE124989/results/GSE124989_VAE-GAN.npy")
    # x = np.load("./Result/GSE147326/GSE147326_VAE+GAN.npy")
    # x = np.load("./Data/GSE147326/results/recovered_data.npy")
    # x = np.load("./Data/GSE147326/GSE147326_log.npy")
    # x = np.load("./Data/GSE271269/GSE271269_VAE+GAN1.npy")
    x = x.astype(np.float64)
    # z = np.load("./Data/GSE123358/GSE123358_VAE+GAN1.npy")
    z = np.load("./Data/GSE123358/GSE123358_log.npy")
    # z = np.load("./Data/GSE124989/GSE124989_log.npy")
    # z = np.load("./Data/GSE147326/results/recovered_data.npy")
    # z = np.load("./Data/GSE271269/GSE271269_log.npy")
    # z = np.load("./Data/GSE124989/GSE124989_log.npy")
    z = z.astype(np.float64)

    print(f"原始数据形状: {x.shape}")
    print(f"原始数据形状: {z.shape}" )

    # 加载基因名称
    all_genes = load_gene_names("./Data/GSE123358/gene.txt")  # 原始数据基因名称
    # all_genes = load_gene_names("./Data/GSE124989/gene.txt")  # 原始数据基因名称
    target_genes = ['BRCA1', 'BRCA2', 'BARD1', 'BRIP1', 'PALB2', 'RAD51', 'RAD54L', 'XRCC3', 'ERBB2', 'ESR1', 'PGR',
                        'PIK3CA', 'TP53', 'PPM1D', 'RB1CC1', 'HMMR', 'NQO2', 'SLC22A18', 'PTEN', 'EGFR', 'KIT', 'NOTCH1',
                        'FZD7', 'LRP6', 'FGFR1', 'CCND1']
    print(f"关键基因匹配情况: {len(map_gene_indices(target_genes, all_genes))}/{len(target_genes)}")

    # 实验参数（新增）
    METHODS = ['pca',  'nmf', 'none']  # 'none'表示不降维
    N_COMPONENTS = 20
    MAX_ITER = 100

    # ================== 修复后的主循环 ==================
    for method in METHODS:
        print(f"\n=== 当前方法: {method.upper()} ===")

        # --- 数据预处理 ---
        if method != 'none':
            reductor = LinearDimReductor(method, N_COMPONENTS)
            processed_x = fortranify(reductor.fit_transform(x))
            print(f"✅ 降维验证 {x.shape} -> {processed_x.shape}")
        else:
            processed_x = fortranify(x)

        # --- 算法初始化 ---
        k = min(int(processed_x.shape[1] * 3), 120)
        NP = 40
        Ws = np.zeros((NP, k, processed_x.shape[1]))
        UW = (np.random.randn(processed_x.shape[0], k).astype(np.float64),
              np.random.randn(k, processed_x.shape[1]).astype(np.float64))

        # --- SMAF初始化 ---
        try:
            UF, WF = smaf(processed_x, k, 5, 0.0001, maxItr=10,
                          UW=UW, donorm=True, mode=1, mink=3.)
            print(f"✅ SMAF初始化成功 UF={UF.shape}, WF={WF.shape}")
        except Exception as e:
            print(f"❌ SMAF失败: {str(e)}")
            continue

        # --- 主优化循环 ---
        maxItr = 100
        fitnessVal = np.zeros((NP, processed_x.shape[1]))

        from tqdm import tqdm

        for gen in tqdm(range(maxItr), desc=f"{method.upper()} 优化进度"):
            # 1. 更新每个样本的权重
            for i in range(processed_x.shape[1]):  # 遍历每个样本
                Ws_tem = Ws[:, :, i]

                # --- IVY算法更新 ---
                fa = IvyAlgorithm(NP, Ws_tem, fitnessVal[:, i], [1.0, 1.0])
                Ws_tem, fitnessVal[:, i] = selection(
                    Ws_tem, fa, fitnessVal[:, i],
                    processed_x[:, i], UF
                )

                # 保存最佳权重
                best_idx = np.argmin(fitnessVal[:, i])
                WF[:, i] = Ws_tem[best_idx]
                Ws[:, :, i] = Ws_tem

            # 2. 更新字典
            UF = spams.lasso(
                fortranify(processed_x.T),
                D=fortranify(WF.T),
                lambda1=0.0005 * (np.linalg.norm(processed_x) ** 2 / processed_x.shape[1]),
                mode=1,
                numThreads=4
            )
            UF = np.asarray(UF.todense()).T.astype(np.float64)

            # 3. 重建数据
            xa_lowdim = UF.dot(WF)
            if method != 'none':
                xa = reductor.inverse_transform(xa_lowdim)
                assert xa.shape == x.shape, f"❌ 维度不匹配 {xa.shape} vs {x.shape}"
            else:
                xa = xa_lowdim

            # 4. 每10次迭代打印进度
            if (gen + 1) % 10 == 0:
                print(f"\n--- Iteration {gen + 1}/{maxItr} ---")
                print(f"当前重建维度: {xa.shape}")
                metrics = main_evaluation(z, xa, all_genes, target_genes)
                print_iteration_summary(gen, metrics)

                # 5. 早停检测
                # ✅ 确保记录所有必要字段
                metrics = main_evaluation(z, xa, all_genes, target_genes)
                result1.append({
                    'iteration': gen + 1,
                    'mse': metrics['mse'],
                    'mae': metrics['mae'],
                    'pcc': metrics['pcc'],
                    'cs': metrics['cs']
                })

                if early_stopping(result1, patience=50):
                    print(f"🏁 早停触发于第 {gen + 1} 次迭代")
                    break

        # --- 最终评估 ---
        final_metrics = main_evaluation(z, xa, all_genes, target_genes)
        print("\n=== 最终结果 ===")
        print_iteration_summary(gen, final_metrics)
        np.savetxt(f'./Result/GSE123358/GSE123358_{method}+PCA.csv', xa, delimiter=',')

    # --- 保存所有结果 ---
    pd.DataFrame(result1).to_csv("D://TCGA-BRCA//DeepCS//Result//GSE123358//123358_PCA-IVYCS.csv", index=False)



