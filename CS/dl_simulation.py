import numpy as np
from sklearn import decomposition
from sklearn.linear_model import MultiTaskLassoCV, OrthogonalMatchingPursuit, RidgeCV, Ridge, ElasticNetCV, Lasso, \
    orthogonal_mp_gram
import spams
from scipy.spatial import distance
from scipy.stats import spearmanr, entropy
import sys
from sklearn import mixture

THREADS = 10


def random_phi(m, g, d_thresh=0.2, nonneg=False):
    #np.random.seed(5)##########
    Phi = np.zeros((m, g))
    Phi[0] = np.random.randn(g)
    if nonneg:
        Phi[0] = abs(Phi[0])
    Phi[0] /= np.linalg.norm(Phi[0])
    for i in range(1, m):
        dmax = 1
        while dmax > d_thresh:
            p = np.random.randn(g)
            if nonneg:
                p = abs(p)
            dmax = max(abs(1 - distance.cdist(Phi, [p], 'correlation')))
        Phi[i] = p / np.linalg.norm(p)
    return Phi


def random_phi_subsets(m, g, n, d_thresh=0.2):
    Phi = np.zeros((m, g))
    Phi[0, np.random.choice(g, n, replace=False)] = n ** -0.5
    for i in range(1, m):
        dmax = 1
        while dmax > d_thresh:
            p = np.zeros(g)
            p[np.random.choice(g, n, replace=False)] = n ** -0.5
            dmax = Phi[:i].dot(p).max()
        Phi[i] = p
    return Phi

# 生成带有噪声的观测数据
def get_observations(X0, Phi, snr=5, return_noise=False):
    #np.random.seed(5)##########
    noise = np.array([np.random.randn(X0.shape[1]) for _ in range(X0.shape[0])])
    noise *= np.linalg.norm(X0) / np.linalg.norm(noise) / snr
    if return_noise:
        return Phi.dot(X0 + noise), noise
    else:
        return Phi.dot(X0 + noise)


def coherence(U, m):
    Phi = random_phi(m, U.shape[0])
    PU = Phi.dot(U)
    d = distance.pdist(PU.T, 'cosine')
    return abs(1 - d)

# 通过OMP算法从观测矩阵Y和感知矩阵D中解码出稀疏信号W
# def sparse_decode(Y, D, k, worstFit=1., mink=4):
#     while k > mink:
#         W = spams.omp(np.asfortranarray(Y), np.asfortranarray(D), L=k, numThreads=THREADS)
#         W = np.asarray(W.todense())
#         fit = 1 - np.linalg.norm(Y - D.dot(W)) ** 2 / np.linalg.norm(Y) ** 2
#         if fit < worstFit:
#             break
#         else:
#             k -= 1
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
# 恢复信号
def recover_system_knownBasis(X0, m, k, Psi=[], use_ridge=False, snr=0, nsr_pool=0, subset_size=0):
    if len(Psi) == 0:
        Psi, s, vt = np.linalg.svd(X0) #svd奇异值分解  Psi为奇异值矩阵
    if subset_size == 0:
        Phi = random_phi(m, X0.shape[0]) #Phi随机测量矩阵
    else:
        Phi = random_phi_subsets(m, X0.shape[0], subset_size)
    Phi_noise = random_phi(m, X0.shape[0]) * nsr_pool  # 生成噪声矩阵
    D = Phi.dot(Psi) #生成字典（投影矩阵）
    Y = get_observations(X0, Phi + Phi_noise, snr=snr)  #生成观测矩阵 模拟了带噪声的测量结果
    W = sparse_decode(Y, D, k)  # 使用D从观测数据Y中解码出稀疏表示W
    if use_ridge:
        X = update_sparse_predictions(Y, D, W, Psi)  # 岭回归更新预测
    else:
        X = Psi.dot(W)
    return X, Phi, Y, W, D, Psi
