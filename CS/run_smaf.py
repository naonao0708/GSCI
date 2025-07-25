import numpy as np
import pandas as pd
import spams
from scipy.stats import entropy
from scipy.spatial import distance
import sys, os
import csv

from sklearn.linear_model import orthogonal_mp_gram

THREADS = 4
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csgraph


# 构建邻接矩阵L，使用k近邻图
def build_laplacian(X, n_neighbors=5):
    # 使用样本作为输入构建邻接矩阵
    A = kneighbors_graph(X, n_neighbors, mode='connectivity', include_self=True)
    L = csgraph.laplacian(A, normed=True)
    return L


# 在 smaf 函数中加入 Laplacian 正则化
def smaf(X, d, lda1, lda2, maxItr=10, UW=None, posW=False, posU=True, use_chol=False, module_lower=500,
         activity_lower=5, donorm=False, mode=1, mink=5, U0=[], U0_delta=0.1, doprint=False, laplacian_reg=0.1):
    # 使用样本作为输入构建拉普拉斯矩阵，使 L 和 U 具有相同的行数
    L = build_laplacian(X, n_neighbors=5)

    if UW is None:
        U, W = spams.nmf(np.asfortranarray(X), return_lasso=True, K=d, numThreads=THREADS)
        W = np.asarray(W.todense())
    else:
        U, W = UW
    Xhat = U.dot(W)
    Xnorm = np.linalg.norm(X) ** 2 / X.shape[1]
    for itr in range(maxItr):
        if mode == 1:
            # 在 Lasso 更新中加入拉普拉斯正则化
            U = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
                            lambda1=lda2 * Xnorm, mode=1, numThreads=THREADS, cholesky=use_chol, pos=posU)
            U = np.asarray(U.todense()).T

            # 确保 L 和 U 维度匹配后，再进行拉普拉斯正则化
            if L.shape[0] == U.shape[0]:
                U += laplacian_reg * L.dot(U)

        elif mode == 2:
            if len(U0) > 0:
                U = projected_grad_desc(W.T, X.T, U.T, U0.T, lda2, U0_delta, maxItr=400)
                U = U.T
            else:
                U = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
                                lambda1=lda2, lambda2=0.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posU)
                U = np.asarray(U.todense()).T

        # if donorm:
        #     U = U / np.linalg.norm(U, axis=0)
        #     U[np.isnan(U)] = 0
        if donorm:
            U_norms = np.linalg.norm(U, axis=0)
            U[:, U_norms > 0] /= U_norms[U_norms > 0]  # 避免除以零
            U[:, U_norms == 0] = 0  # 对零向量列赋零值
            U[np.isnan(U)] = 0

        if mode == 1:
            wf = (1 - lda2)
            W = sparse_decode(X, U, max(lda1, mink + 1), worstFit=wf, mink=mink)
        elif mode == 2:
            if len(U0) > 0:
                W = projected_grad_desc(U, X, W, [], lda1, 0., nonneg=posW, maxItr=400)
            else:
                W = spams.lasso(np.asfortranarray(X), D=np.asfortranarray(U),
                                lambda1=lda1, lambda2=1.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posW)
                W = np.asarray(W.todense())

        Xhat = U.dot(W)
        module_size = np.average([np.exp(entropy(u)) for u in U.T if u.sum() > 0])
        activity_size = np.average([np.exp(entropy(abs(w))) for w in W.T])

        if doprint:
            print(distance.correlation(X.flatten(), Xhat.flatten()), module_size, activity_size, lda1, lda2)
        if module_size < module_lower:
            lda2 /= 2.
        if activity_size < activity_lower:
            lda2 /= 2.

    return U, W
# 在给定的字典 D下，找到输入数据 Y 的最佳稀疏表示 W，通过逐步减少允许的非零元素数量来优化拟合度，直到达到一定的拟合标准或最小稀疏度限制。
# def sparse_decode(Y, D, k, worstFit=1., mink=4):
#     # 检查初始条件
#     if k <= mink:
#         raise ValueError(f"k ({k}) must be greater than mink ({mink}). Adjust your input parameters.")
#     W = None
#     # k当前选择的稀疏表示中非零元素
#     while k > mink:
#         W = spams.omp(np.asfortranarray(Y), np.asfortranarray(D), L=k, numThreads=THREADS)
#         W = np.asarray(W.todense())
#         fit = 1 - np.linalg.norm(Y - D.dot(W)) ** 2 / np.linalg.norm(Y) ** 2 #计算重构误差来评估拟合度
#         if fit < worstFit:
#             break
#         else:
#             k -= 1 #如果拟合度足够好，减少 k 的值，尝试更少的非零元素，以寻找更好的稀疏表示。
#         # 确保 W 已定义
#         if W is None:
#             W = np.zeros((D.shape[1], Y.shape[1]))  # 若未定义则初始化为零矩阵
#             print("Warning: sparse_decode exited without valid W. Returned a zero matrix.")
#     return W
def sparse_decode(Y, D, k, worstFit=1., mink=4):
    """
    此函数使用压缩感知正交匹配追踪（CoSaMP）算法对输入信号 Y 进行稀疏分解
    :param Y: 输入信号矩阵
    :param D: 字典矩阵
    :param k: 初始迭代次数
    :param worstFit: 拟合的最差阈值
    :param mink: 最小迭代次数
    :return: 稀疏表示矩阵 W
    """
    Gram = D.T.dot(D)
    Dy = D.T.dot(Y)
    while k > mink:
        # 使用 sklearn 的 orthogonal_mp_gram 实现 CoSaMP 算法
        W = orthogonal_mp_gram(Gram, Dy, n_nonzero_coefs=k)
        fit = 1 - np.linalg.norm(Y - D.dot(W)) ** 2 / np.linalg.norm(Y) ** 2
        if fit < worstFit:
            break
        else:
            k -= 1
    return W

# ##### 构建高斯相似性拉普拉斯矩阵 #########
# from sklearn.metrics.pairwise import rbf_kernel
# from sklearn.neighbors import kneighbors_graph
# from scipy.sparse import csgraph
# from scipy.stats import entropy
# import numpy as np
# import spams
#
#
#
# # 构建高斯相似性拉普拉斯矩阵
# def build_laplacian(X, n_neighbors=5, gamma=0.5, use_gaussian=False):
#     """
#     构建邻接矩阵并生成拉普拉斯矩阵
#     :param X: 数据矩阵
#     :param n_neighbors: k近邻数量
#     :param gamma: 高斯相似性核参数
#     :param use_gaussian: 是否使用高斯相似性代替k近邻
#     :return: 拉普拉斯矩阵
#     """
#     if use_gaussian:
#         # 高斯相似性邻接矩阵
#         A = rbf_kernel(X, gamma=gamma)
#     else:
#         # k近邻邻接矩阵
#         A = kneighbors_graph(X, n_neighbors, mode='connectivity', include_self=True)
#     # 拉普拉斯矩阵
#     L = csgraph.laplacian(A, normed=True)
#     return L
# def dynamic_regularization(lda1, lda2, itr, maxItr):
#     # 正则化参数随迭代减小
#     lda1_new = lda1 * (1 - itr / maxItr)
#     lda2_new = lda2 * (1 - itr / maxItr)
#     return lda1_new, lda2_new
#
#
# # 在 smaf 函数中加入拉普拉斯正则化
# def smaf(X, d, lda1, lda2, maxItr=10, UW=None, posW=False, posU=True, use_chol=False, module_lower=500,
#          activity_lower=5, donorm=False, mode=1, mink=5, U0=[], U0_delta=0.1, doprint=False,
#          laplacian_reg=0.1, dynamic_reg=True, use_gaussian=False):
#     """
#     稀疏矩阵分解算法 (加入拉普拉斯正则化和动态调整)
#     :param X: 输入数据矩阵
#     :param d: 稀疏基数量
#     :param lda1: L1正则化参数
#     :param lda2: L2正则化参数
#     :param maxItr: 最大迭代次数
#     :param UW: 初始分解 (U, W)
#     :param posW: W的非负性约束
#     :param posU: U的非负性约束
#     :param use_chol: 是否使用Cholesky分解加速
#     :param module_lower: 模块稀疏性下限
#     :param activity_lower: 活跃稀疏性下限
#     :param donorm: 是否归一化U
#     :param mode: 分解模式
#     :param mink: 最小稀疏度
#     :param laplacian_reg: 拉普拉斯正则化权重
#     :param dynamic_reg: 是否动态调整正则化参数
#     :param use_gaussian: 是否使用高斯相似性
#     :return: 分解后的U, W
#     """
#     # 构建拉普拉斯矩阵
#     L = build_laplacian(X, n_neighbors=5, gamma=0.5, use_gaussian=use_gaussian)
#
#     if UW is None:
#         U, W = spams.nmf(np.asfortranarray(X), return_lasso=True, K=d, numThreads=THREADS)
#         W = np.asarray(W.todense())
#     else:
#         U, W = UW
#
#     Xnorm = np.linalg.norm(X) ** 2 / X.shape[1]
#
#     for itr in range(maxItr):
#         # 动态调整正则化参数
#         if dynamic_reg:
#             lda1, lda2 = dynamic_regularization(lda1, lda2, itr, maxItr)
#
#         if mode == 1:
#             # Lasso更新U，加入拉普拉斯正则化
#             U_update = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
#                                    lambda1=lda2 * Xnorm, mode=1, numThreads=THREADS, cholesky=use_chol, pos=posU)
#             U_update = np.asarray(U_update.todense()).T
#
#             # 拉普拉斯正则化调整
#             if L.shape[0] == U.shape[0]:
#                 U_update += laplacian_reg * L.dot(U_update)
#
#             # 动量更新
#             momentum = 0.9
#             U = momentum * U + (1 - momentum) * U_update
#
#         elif mode == 2:
#             if len(U0) > 0:
#                 U = projected_grad_desc(W.T, X.T, U.T, U0.T, lda2, U0_delta, maxItr=400)
#                 U = U.T
#             else:
#                 U = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
#                                 lambda1=lda2, lambda2=0.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posU)
#                 U = np.asarray(U.todense()).T
#
#         # 归一化U
#         if donorm:
#             U_norms = np.linalg.norm(U, axis=0)
#             U[:, U_norms > 0] /= U_norms[U_norms > 0]
#             U[:, U_norms == 0] = 0
#             U[np.isnan(U)] = 0
#
#         # 更新W
#         if mode == 1:
#             wf = (1 - lda2)
#             W = sparse_decode(X, U, max(lda1, mink + 1), worstFit=wf, mink=mink)
#         elif mode == 2:
#             if len(U0) > 0:
#                 W = projected_grad_desc(U, X, W, [], lda1, 0., nonneg=posW, maxItr=400)
#             else:
#                 W = spams.lasso(np.asfortranarray(X), D=np.asfortranarray(U),
#                                 lambda1=lda1, lambda2=1.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posW)
#                 W = np.asarray(W.todense())
#
#         # 稀疏性评估
#         module_size = np.average([np.exp(entropy(u)) for u in U.T if u.sum() > 0])
#         activity_size = np.average([np.exp(entropy(abs(w))) for w in W.T])
#
#         if doprint:
#             print(f"Iteration {itr + 1}: Module sparsity = {module_size}, Activity sparsity = {activity_size}")
#         if module_size < module_lower:
#             lda2 /= 2.
#         if activity_size < activity_lower:
#             lda2 /= 2.
#
#     return U, W
# # 在给定的字典 D下，找到输入数据 Y 的最佳稀疏表示 W，通过逐步减少允许的非零元素数量来优化拟合度，直到达到一定的拟合标准或最小稀疏度限制。
# def sparse_decode(Y, D, k, worstFit=1., mink=4):
#     # 检查初始条件
#     if k <= mink:
#         raise ValueError(f"k ({k}) must be greater than mink ({mink}). Adjust your input parameters.")
#     W = None
#     # k当前选择的稀疏表示中非零元素
#     while k > mink:
#         W = spams.omp(np.asfortranarray(Y), np.asfortranarray(D), L=k, numThreads=THREADS)
#         W = np.asarray(W.todense())
#         fit = 1 - np.linalg.norm(Y - D.dot(W)) ** 2 / np.linalg.norm(Y) ** 2 #计算重构误差来评估拟合度
#         if fit < worstFit:
#             break
#         else:
#             k -= 1 #如果拟合度足够好，减少 k 的值，尝试更少的非零元素，以寻找更好的稀疏表示。
#         # 确保 W 已定义
#         if W is None:
#             W = np.zeros((D.shape[1], Y.shape[1]))  # 若未定义则初始化为零矩阵
#             print("Warning: sparse_decode exited without valid W. Returned a zero matrix.")
#     return W




# 使用稀疏非负矩阵分解NMF算法进行矩阵分解 X≈UW
# def smaf(X, d, lda1, lda2, maxItr=10, UW=None, posW=False, posU=True, use_chol=False, module_lower=500,
#          activity_lower=5, donorm=False, mode=1, mink=5, U0=[], U0_delta=0.1, doprint=False):
#     # use Cholesky when we expect a very sparse result
#     # this tends to happen more on the full vs subsampled matrices
#     if UW == None:
#         U, W = spams.nmf(np.asfortranarray(X), return_lasso=True, K=d, numThreads=THREADS)
#         W = np.asarray(W.todense())  # 矩阵分解，并将结果转换为稠密格式
#     else:
#         U, W = UW
#     Xhat = U.dot(W) # 计算重构矩阵
#     Xnorm = np.linalg.norm(X) ** 2 / X.shape[1]  #计算X的范数
#     for itr in range(maxItr):
#         if mode == 1:
#             # In this mode the ldas correspond to an approximate desired fit
#             # Higher lda will be a worse fit, but will result in a sparser sol'n
#             U = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
#                             lambda1=lda2 * Xnorm, mode=1, numThreads=THREADS, cholesky=use_chol, pos=posU)
#             U = np.asarray(U.todense()).T #使用Lasso通过X和W更新U，参数调整使得稀疏性更高
#         elif mode == 2:
#             if len(U0) > 0:
#                 U = projected_grad_desc(W.T, X.T, U.T, U0.T, lda2, U0_delta, maxItr=400) # 使用投影梯度下降法优化U
#                 U = U.T
#             else: # 如果没有U0，使用Lasso更新U
#                 U = spams.lasso(np.asfortranarray(X.T), D=np.asfortranarray(W.T),
#                                 lambda1=lda2, lambda2=0.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posU)
#                 U = np.asarray(U.todense()).T
#         if donorm:
#             U = U / np.linalg.norm(U, axis=0) # 归一化
#             U[np.isnan(U)] = 0 # 处理NaN值
#         if mode == 1:
#             wf = (1 - lda2)
#             W = sparse_decode(X, U, lda1, worstFit=wf, mink=mink)  # 通过稀疏编码的方式更新 W
#         elif mode == 2:
#             if len(U0) > 0:
#                 W = projected_grad_desc(U, X, W, [], lda1, 0., nonneg=posW, maxItr=400) # 使用投影下降方法更新 W
#             else:
#                 W = spams.lasso(np.asfortranarray(X), D=np.asfortranarray(U),
#                                 lambda1=lda1, lambda2=1.0, mode=2, numThreads=THREADS, cholesky=use_chol, pos=posW)
#                 W = np.asarray(W.todense())
#         Xhat = U.dot(W) # 计算重构矩阵
#         module_size = np.average([np.exp(entropy(u)) for u in U.T if u.sum() > 0]) # 计算模块大小：通过计算U的每一列的熵值来估算模块的稀疏程度
#         activity_size = np.average([np.exp(entropy(abs(w))) for w in W.T]) #计算活动大小，通过熵值估算W的稀疏程度
#         if doprint:
#             print(distance.correlation(X.flatten(), Xhat.flatten()), module_size, activity_size, lda1, lda2)
#         if module_size < module_lower:
#             lda2 /= 2.
#         if activity_size < activity_lower:
#             lda2 /= 2.
#     return U, W


