import numpy as np

# 加载数据
data = np.load('./Data/GSE123358/GSE123358.npy')

# Step 1: 计算稀疏元素的比例
# 计算数据中为零的元素数量
sparse_elements = np.sum(data == 0)

# 计算数据中的总元素数量
total_elements = data.size

# 计算稀疏比例
sparsity_ratio = sparse_elements / total_elements

# 输出稀疏比例
print(f"稀疏比例: {sparsity_ratio:.4f}")

