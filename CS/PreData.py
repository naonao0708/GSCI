import pandas as pd
import numpy as np

# 读取原始数据
# data = pd.read_csv("./Data/GSE123358/GSE123358.csv", index_col=0)
# data = pd.read_csv("./Data/GSE271269/GSE271269.csv", index_col=0)
# data = pd.read_csv("./Data/GSE124989/GSE124989_annotated.csv", index_col=0)

# data = pd.read_csv("./Data/GSE147326/GSE147326_fill.csv", index_col=0)
data = pd.read_excel("./Data/GSE22219.xlsx", engine='openpyxl')
# data = pd.read_excel("./Data/GSE124989/GSE124989.xlsx", engine='openpyxl')
# 过滤细胞：保留表达值大于等于200的细胞
cell_filter = data.sum(axis=0) >= 200  # 对每列（细胞）求和，保留表达值大于等于200的细胞
filtered_data_cells = data.loc[:, cell_filter]

# 过滤基因：保留表达值大于等于3的基因
gene_filter = filtered_data_cells.sum(axis=1) >= 3  # 对每行（基因）求和，保留表达值大于等于3的基因
filtered_data_genes = filtered_data_cells.loc[gene_filter, :]
# np.save("./Data/GSE123358/GSE123358.npy", filtered_data_genes.values)
# filtered_data_genes.to_csv("./Data/GSE124989/GSE124989_raw.csv")
# np.save("./Data/GSE124989/GSE124989_raw.npy", filtered_data_genes.values)
# filtered_data_genes.to_csv("./Data/GSE123358/GSE123358_raw.csv")
# np.save("./Data/GSE123358/GSE123358_raw.npy", filtered_data_genes.values)
# filtered_data_genes.to_csv("./Data/GSE147326/GSE147326_raw.csv")
# np.save("./Data/GSE147326/GSE147326_raw.npy", filtered_data_genes.values)
filtered_data_genes.to_csv("./Data/GSE22219_raw.csv")
np.save("./Data/GSE22219_raw.npy", filtered_data_genes.values)
#
# 对数归一化（log transformation）
# 添加一个常数1，避免log(0)的情况
log_normalized_data = np.log1p(filtered_data_genes)

# 保存为 CSV 文件（保留行名和列名）
# filtered_data_genes.to_csv("./Data/GSE271269/GSE271269_log.csv")
# log_normalized_data.to_csv("./Data/GSE147326/GSE147326_log.csv")
log_normalized_data.to_csv("./Data/GSE22219_log.csv")
# log_normalized_data.to_csv("./Data/GSE124989/GSE124989_log.csv")
# log_normalized_data.to_csv("./Data/GSE123358/GSE123358_log.csv")

# 保存为 Numpy 文件（仅保存数据）
# np.save("./Data/GSE123358/GSE123358_log.npy", log_normalized_data.values)
np.save("./Data/GSE22219_log.npy", log_normalized_data.values)
# np.save("./Data/GSE124989/GSE124989_log.npy", log_normalized_data.values)
