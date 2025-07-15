
import matplotlib.pyplot as plt
import numpy as np
# import cv2
import os

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# from hparams import drive_fernanda

def plot_linhas(obj, axis):
  # axis.plot((obj.ariri[:, 0] + 1)*obj.resolucao, obj.ariri[:, 1], color = 'cyan', linewidth = 7)
  axis.plot((obj.barravelha[:, 0] + 1)*obj.resolucao, obj.barravelha[:, 1], color = 'lime', linewidth = 7)
  axis.plot((obj.camboriu[:, 0] + 1)*obj.resolucao, obj.camboriu[:, 1], color = 'blueviolet', linewidth = 7)

def plot(obj, atributo, color = 'seismic', linhas = False, zoom = None, percentil = (1, 99)):
    plt.figure(figsize = [24, 10])
    if zoom is not None:
        plt.ylim(zoom)

    vmin = np.percentile(atributo, percentil[0])
    vm = np.percentile(atributo, percentil[1])
    plt.imshow(atributo.T, cmap = color,
                extent = obj.extent, aspect = 'auto', vmin = vmin, vmax = vm)
    if linhas:
        plot_linhas(obj, plt)
    plt.show()

def subplot(obj, atributos, colors = ['tab10', 'tab10'], zoom = None, percentils = [(0, 100), (0, 100)]):
    fig, axes = plt.subplots(2, 1, figsize = [24, 20])
    if zoom is not None:
        [ax.set_ylim(zoom) for ax in axes]

    for i in range(2):
        vmin = np.percentile(atributos[i], percentils[i][0])
        vm = np.percentile(atributos[i], percentils[i][1])
        axes[i].imshow(atributos[i].T, cmap = colors[i],
                    extent = obj.extent, aspect = 'auto', vmin = vmin, vmax = vm)

def concatenate(atributos):
    X = np.copy(atributos[0]).reshape(-1, 1)
    for atributo in atributos[1:]:
        X = np.concatenate((X, atributo.reshape(-1, 1)), axis = 1)
    return X

cores = np.array([ # padrao png
    [44, 44, 160], # background vermelho
    [0, 102, 255], # laranja
    [255, 255, 0], # azul claro
    [170, 170, 255], # bege
    [44,  90, 160], # marrom
    [44, 160,  90], # verde
    [68,  0, 85], # roxo
])

# def colore(obj, path, cores = cores):
#     img = cv2.imread(os.path.join(drive_fernanda, path))
    
#     m, n = obj.X.T.shape
#     M, N = img.shape[:-1]
#     i = np.linspace(start = 0, stop = M - 1, num = m, dtype = int)
#     j = np.linspace(start = 0, stop = N - 1, num = n, dtype = int)

#     downsample = img[i, :]
#     downsample = downsample[:, j]

#     color = np.zeros(shape = (m, n))
#     for i in tqdm(range(m)):
#         for j in range(n):
#             color[i, j] = np.argmin(np.linalg.norm(downsample[i, j, :] - cores, axis = 1))
#     color = color.T
    
#     return color

def kmenas_ablation(atributos, k = 4, seed = 1024):
  X = concatenate(atributos)

  scaler = StandardScaler()
  Z = scaler.fit_transform(X)
  pca = PCA()
  H = pca.fit_transform(Z)

  kmenas = KMeans(n_clusters = k, n_init = 'auto', random_state = seed).fit(H)
  clusters = kmenas.labels_

  # enc = OneHotEncoder()
  # onehot = enc.fit_transform(clusters.reshape(-1, 1)).toarray()
  # onehot = onehot.reshape([atributos[0].shape[0], atributos[0].shape[1], -1])

  clusters = clusters.reshape(atributos[0].shape)
  return clusters

def segmenta(obj):
    extent = obj.extent[2:]
    barravelha = obj.barravelha[:, 1]
    barravelha = (barravelha - extent[-1]) / (extent[-2] - extent[-1])
    camboriu = obj.camboriu[:, 1]
    camboriu = (camboriu - extent[-1]) / (extent[-2] - extent[-1])

    segmentado = np.zeros(shape = obj.X.shape)
    x = segmentado.shape[0]
    h = segmentado.shape[1]
    subsample = np.linspace(0, barravelha.shape[0] - 1, x, dtype = int)
    for j in range(x):
        coluna = np.zeros(shape = segmentado[j, :].shape)
        b = int(barravelha[subsample[j]] * h)
        c = int(camboriu[subsample[j]] * h)
        coluna[:b] = 0
        coluna[b:c] = 1
        coluna[c:] = 0
        segmentado[j, :] = coluna
    obj.segmentado = segmentado