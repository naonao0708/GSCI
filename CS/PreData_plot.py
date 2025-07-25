# import pandas as pd
# import numpy as np
#
# # 读取原始数据
# # data = pd.read_csv("./Data/GSE123358/GSE123358.csv", index_col=0)
# # data = pd.read_csv("./Data/GSE271269/GSE271269.csv", index_col=0)
# # data = pd.read_csv("./Data/GSE124989/GSE124989_annotated.csv", index_col=0)
#
# data = pd.read_csv("./Data/GSE147326/GSE147326_fill.csv", index_col=0)
# # data = pd.read_excel("./Data/GSE124989/GSE124989.xlsx", engine='openpyxl')
# # 过滤细胞：保留表达值大于等于200的细胞
# cell_filter = data.sum(axis=0) >= 200  # 对每列（细胞）求和，保留表达值大于等于200的细胞
# filtered_data_cells = data.loc[:, cell_filter]
#
# # 过滤基因：保留表达值大于等于3的基因
# gene_filter = filtered_data_cells.sum(axis=1) >= 3  # 对每行（基因）求和，保留表达值大于等于3的基因
# filtered_data_genes = filtered_data_cells.loc[gene_filter, :]
# # np.save("./Data/GSE123358/GSE123358.npy", filtered_data_genes.values)
# # filtered_data_genes.to_csv("./Data/GSE124989/GSE124989_raw.csv")
# # np.save("./Data/GSE124989/GSE124989_raw.npy", filtered_data_genes.values)
# # filtered_data_genes.to_csv("./Data/GSE123358/GSE123358_raw.csv")
# # np.save("./Data/GSE123358/GSE123358_raw.npy", filtered_data_genes.values)
# filtered_data_genes.to_csv("./Data/GSE147326/GSE147326_raw.csv")
# np.save("./Data/GSE147326/GSE147326_raw.npy", filtered_data_genes.values)
# #
# # 对数归一化（log transformation）
# # 添加一个常数1，避免log(0)的情况
# log_normalized_data = np.log1p(filtered_data_genes)
#
# # 保存为 CSV 文件（保留行名和列名）
# # filtered_data_genes.to_csv("./Data/GSE271269/GSE271269_log.csv")
# log_normalized_data.to_csv("./Data/GSE147326/GSE147326_log.csv")
# # log_normalized_data.to_csv("./Data/GSE124989/GSE124989_log.csv")
# # log_normalized_data.to_csv("./Data/GSE123358/GSE123358_log.csv")
#
# # 保存为 Numpy 文件（仅保存数据）
# # np.save("./Data/GSE123358/GSE123358_log.npy", log_normalized_data.values)
# np.save("./Data/GSE147326/GSE147326_log.npy", log_normalized_data.values)
# # np.save("./Data/GSE124989/GSE124989_log.npy", log_normalized_data.values)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# === Step 1: 配置绘图风格 ===
sns.set(style="whitegrid")
rcParams['font.family'] = 'Times New Roman'
rcParams['font.size'] = 12

# === Step 2: 读取原始数据 ===
raw_data = pd.read_csv("./Data/GSE147326/GSE147326_fill.csv", index_col=0)
# raw_data = pd.read_excel("./Data/GSE124989/GSE124989.xlsx", engine='openpyxl')
# raw_data = raw_data.apply(pd.to_numeric, errors='coerce')  # 转为 float，非数值变 NaN
# raw_data = raw_data.fillna(0)  # 将 NaN 替换为 0，避免后续出错
# raw_data = pd.read_csv("./Data/GSE123358/GSE123358.csv", index_col=0)

# === Step 3: 预处理操作（过滤+归一化） ===
# 过滤细胞：表达总和 ≥ 200
cell_filter = raw_data.sum(axis=0) >= 200
filtered_cells = raw_data.loc[:, cell_filter]

# 过滤基因：表达总和 ≥ 3
gene_filter = filtered_cells.sum(axis=1) >= 3
filtered_data = filtered_cells.loc[gene_filter, :]

# 对数归一化（log1p）
log_data = np.log1p(filtered_data)

# === Step 4: 计算统计量 ===
def get_qc_stats(df):
    total_counts = df.sum(axis=0)  # 每个细胞的表达总量
    n_genes = (df > 0).sum(axis=0)  # 每个细胞表达的基因数
    return total_counts, n_genes

raw_counts, raw_genes = get_qc_stats(raw_data)
proc_counts, proc_genes = get_qc_stats(log_data)

# # === Step 5: 绘图 ===
# fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#
# # (a) 原始数据
# axs[0].scatter(raw_counts, raw_genes, s=5, alpha=0.5)
# # axs[0].set_title("Before Filtering + Normalization")
# axs[0].set_xlabel("Total Counts")
# axs[0].set_ylabel("Number of Genes (non-zero)")
# axs[0].grid(True)
#
# # (b) 预处理后数据
# axs[1].scatter(proc_counts, proc_genes, s=5, alpha=0.5, color="orange")
# # axs[1].set_title("After Filtering + Normalization")
# axs[1].set_xlabel("Total Counts")
# axs[1].set_ylabel("Number of Genes (non-zero)")
# axs[1].grid(True)
#
# plt.tight_layout()
# plt.show()


# === Step 5: 绘图 ===
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# 左图：预处理前
axs[0].scatter(raw_counts, raw_genes, s=10, alpha=0.5)
# axs[0].set_title("Before Filtering + Normalization", fontsize=14)
axs[0].set_xlabel("Total Counts", fontsize=12)
axs[0].set_ylabel("Number of Genes", fontsize=12)
axs[0].tick_params(labelsize=10)
axs[0].text(0.95, 0.05, f"n={raw_data.shape[1]}", transform=axs[0].transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

# 右图：预处理后
axs[1].scatter(proc_counts, proc_genes, s=10, alpha=0.5, color="orange")
# axs[1].set_title("After Filtering + Normalization", fontsize=14)
axs[1].set_xlabel("Total Counts", fontsize=12)
axs[1].set_ylabel("Number of Genes", fontsize=12)
axs[1].tick_params(labelsize=10)
axs[1].text(0.95, 0.05, f"n={log_data.shape[1]}", transform=axs[1].transAxes,
            ha='right', va='bottom', fontsize=10, color='gray')

plt.tight_layout()
axs[0].set_ylim(bottom=0)
axs[1].set_ylim(bottom=0)
axs[0].set_xlim(left=0)
axs[1].set_xlim(left=0)

# 隐藏 x轴第一个0
axs[0].get_xaxis().get_major_ticks()[0].label1.set_visible(False)
axs[1].get_xaxis().get_major_ticks()[0].label1.set_visible(False)



# === Step 6: 保存图像（PDF 高清）===
plt.savefig("QC_Before_After_GSE147326.pdf", dpi=300, bbox_inches='tight')
# 可选 PNG
# plt.savefig("QC_Before_After_GSE147326.png", dpi=300, bbox_inches='tight')

plt.show()