import numpy as np
import spams
import sys, getopt
import pandas as pd

from dl_simulation import *
# from signature_genes import get_signature_genes, build_signature_model
from analyze_predictions import *
# from union_of_transforms import random_submatrix, double_sparse_nmf, smaf
from union_of_transforms import random_submatrix, smaf


def compare_results(A, B):
    results = list(correlations(A, B, 0))[:-1]
    results += list(compare_distances(A, B))
    results += list(compare_distances(A.T, B.T))
    # results += [compare_clusters(A,B)]
    # results += [compare_clusters(A.T,B.T)]
    return results


THREADS = 10

# Results will consist of the following for each method (in both training and testing):
# overall Pearson, overall Spearman, gene Pearson, sample Pearson, sample dist Pearson, sample dist Spearman, gene dist Pearson, gene dist Spearman, sample cluster MI, gene cluster MI

if __name__ == "__main__":
    biased_training = 0.
    composition_noise = 0.
    subset_size = 0
    # opts, args = getopt.getopt(sys.argv[1:], 'i:m:s:d:t:g:n:r:b:a:', [])
    filename = {"data1": "GSE71858.npy", #
                "data2": "GSE60361.npy", #
                "data3": "GSE62270.npy", #
                "data4": "GSE48968.npy", #
                "data5": "GSE52529.npy", #
                "data6": "GSE77564.npy",
                "data7": "GSE78779.npy", #
                "data8": "GSE10247.npy", #
                "data9": "GSE69405.npy"}
    # data_path,measurements,sparsity,dictionary_size,training_dictionary_fraction,max_genes,max_samples,SNR,biased_training
    # = 'GTEx/data.commonGenes.npy', 100, 15, 0.5, 0.05, 5000, 10000, 2., 0.
    opts = {
        '-i': "./Data/GSE147326/GSE147326_fill.csv",
        '-m': '50',
        '-s': '10',
        '-d': '0.5',
        '-t': '0.7',
        '-g': '5000',
        '-n': '200',
        '-r': '2.',
        '-b': '0.',
        # '-a': '',
        # '-z': ''
    }
    for opt, arg in opts.items():
        if opt == '-i':
            data_path = arg
        elif opt == '-m':
            measurements = int(arg)
        elif opt == '-s':
            sparsity = int(arg)
        elif opt == '-d':
            dictionary_size = float(arg)
        elif opt == '-t':
            training_dictionary_fraction = float(arg)
        elif opt == '-g':
            max_genes = int(arg)
        elif opt == '-n':
            max_samples = int(arg)
        elif opt == '-r':
            SNR = float(arg)
        elif opt == '-b':
            biased_training = float(arg)
        elif opt == '-a':
            composition_noise = float(arg)
        elif opt == '-z':
            subset_size = int(arg)
    # # data_path,measurements,sparsity,dictionary_size,training_dictionary_fraction,max_genes,max_samples,SNR,biased_training = 'GTEx/data.commonGenes.npy',100,15,0.5,0.05,5000,10000,2.,0.
    # X = np.load(data_path)
    # print(X.shape)
    # #X0, xo, Xobs = random_submatrix(X, max_genes, max_samples, 0)
    # X0 = X
    # #np.save("./Data/linedata/GSE71858_X0.npy", X0)
    # #np.save("./Data/linedata/GSE77564_X0.npy", X0)

    # 读取CSV数据
    # df = pd.read_csv('./Data/GSE271269.csv', index_col=0)
    # 读取 Excel 文件，index_col=0 表示将第一列作为索引
    # df = pd.read_excel('./Data/GSE140494.xlsx', index_col=0, engine='openpyxl')
    # df = pd.read_csv('./Data/TCGA_BRCA_copy.tsv', index_col=0, sep='\t')
    # df = pd.read_excel('./Data/GSE123358.xlsx', index_col=0, engine='openpyxl')
    # df = pd.read_excel('./Data/GSE124989.xlsx', index_col=0, engine='openpyxl')
    df = pd.read_csv('./Data/GSE147326/GSE147326_fill.csv', index_col=0)
    X = df.values
    print(X.shape)
    X0 = X

    # train bases
    training_dictionary_size = max(int(training_dictionary_fraction * X0.shape[1]), 5)
    if dictionary_size < 1:
        dictionary_size = dictionary_size * training_dictionary_size
    dictionary_size = int(dictionary_size)
    xi = np.zeros(X0.shape[1], dtype=bool)
    if biased_training > 0:
        i = np.random.randint(len(xi))
        dist = distance.cdist([X0[:, i]], X0.T, 'correlation')[0]
        didx = np.argsort(dist)[1:int(biased_training * training_dictionary_size) + 1]
    else:
        didx = []
    xi[didx] = True
    if biased_training < 1:
        remaining_idx = np.setdiff1d(range(len(xi)), didx)
        xi[np.random.choice(remaining_idx, training_dictionary_size - xi.sum(), replace=False)] = True
    xa = X0[:, xi]
    xb = X0[:, np.invert(xi)]
    print('data: %s measurements: %d, sparsity: %d, dictionary size: %d, training fraction: %.2f, genes: %d, samples: %d, SNR: %.1f, bias: %.1f, composition_noise: %.2f, subset_size: %d' % (
        data_path,
        measurements, sparsity, dictionary_size, training_dictionary_fraction, X0.shape[0], X0.shape[1], SNR,
        biased_training, composition_noise, subset_size))
    Results = {}

    # np.save("./Data/linedata/GSE78779_xa.npy", xa)
    # np.save("./Data/linedata/GSE78779_xb.npy", xb)
    # np.save("Data/linedata/GSE123358_xa.npy", xa)
    # np.save("Data/linedata/GSE123358_xb.npy", xb)
    # np.save("Data/linedata/GSE124989_xa.npy", xa)
    # np.save("Data/linedata/GSE124989_xb.npy", xb)
    np.save("Data/GSE147326/GSE147326_xa.npy", xa)
    np.save("Data/GSE147326/GSE147326_xb.npy", xb)

    # ua, sa, vta = np.linalg.svd(xa, full_matrices=False)
    # ua = ua[:, :min(dictionary_size, xa.shape[1])]
    # x1a, phi, y, w, d, psi = recover_system_knownBasis(xa, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False)
    # Results['SVD (training)'] = compare_results(xa, x1a)
    # x1b, phi, y, w, d, psi = recover_system_knownBasis(xb, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False)
    # Results['SVD (testing)'] = compare_results(xb, x1b)
    # np.save("./Data/tcga_gene_svd.npy", x1b)
    #
    #
    # ua, va = spams.nmf(np.asfortranarray(xa), return_lasso=True, K=dictionary_size, clean=True, numThreads=THREADS)
    # x2a, phi, y, w, d, psi = recover_system_knownBasis(xa, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False)
    # Results['sparse NMF (training)'] = compare_results(xa, x2a)
    # x2b, phi, y, w, d, psi = recover_system_knownBasis(xb, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False)
    #
    # Results['sparse NMF (testing)'] = compare_results(xb, x2b)
    # np.save("./Data/tcga_gene_snmf.npy", x2b)
    #
    # k = min(int(xa.shape[1] * 3), 150)
    # #k = 220
    # UW = (np.random.random((xa.shape[0], k)), np.random.random((k, xa.shape[1])))
    # ua, va = smaf(xa, k, 5, 0.0005, maxItr=10, use_chol=True, activity_lower=0., module_lower=xa.shape[0] / 10, UW=UW,
    #               donorm=True, mode=1, mink=3.)
    #
    # x2a, phi, y, w, d, psi = recover_system_knownBasis(xa, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False)
    # Results['SMAF (training)'] = compare_results(xa, x2a)
    # x2b, phi, y, w, d, psi = recover_system_knownBasis(xb, measurements, sparsity, Psi=ua, snr=SNR, use_ridge=False,
    #                                                    nsr_pool=composition_noise, subset_size=subset_size)
    # Results['SMAF (testing)'] = compare_results(xb, x2b)
    # np.save("./Data/tcga_gene_smaf.npy", x2b)
    #
    # print (ua.shape, X0.shape, xa.shape, xb.shape)
    # #print w.shape, w
    # print(y.shape)
    # for k, v in sorted(Results.items()):
    #     print('\t'.join([k] + [str(x) for x in v]))
