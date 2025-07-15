
import numpy as np
from scipy.signal import hilbert
import scipy.ndimage
import segyio

def complexo(X):
  envelope = np.zeros(shape = X.shape)
  fase = np.zeros(shape = X.shape)

  for i in range(X.shape[0]):
    signal = hilbert(X[i, :])

    envelope[i, :] = np.abs(signal)
    fase[i, :] = np.unwrap(np.angle(signal))
  freq = np.diff(fase) / 2 / np.pi
  freq = np.hstack((freq[:, 0:1], freq))
  sweetness = envelope / np.sqrt(np.abs(freq + 1e-6))

  return envelope, fase, freq, sweetness

def coordenadas(X):
  xx, yy = np.indices(X.shape)
  xx = xx / X.shape[0]
  yy = yy / X.shape[1]

  return xx, yy

def coherence(X):
  gersztenkorn = _moving_window(X, _gersztenkorn, (3, 3))
  gersztenkorn = np.nan_to_num(gersztenkorn)

  sobel = np.sqrt(sum(scipy.ndimage.sobel(gersztenkorn, axis) ** 2 for axis in range(2)))

  marfurt = _moving_window(X, _marfurt, (3, 3))
  marfurt = np.nan_to_num(marfurt)

  return gersztenkorn, sobel, marfurt

def _moving_window(data, func, window):
  wrapped = lambda x: func(x.reshape(window))
  return scipy.ndimage.generic_filter(data, wrapped, window)

def _gersztenkorn(region):
  region = region.reshape(-1, region.shape[-1])
  cov = region.dot(region.T)
  vals = np.linalg.eigvalsh(cov)
  return vals.max() / vals.sum()

def _marfurt(region):
  region = region.reshape(-1, region.shape[-1])
  ntraces, nsamples = region.shape
  cov = region.dot(region.T)
  sembl = cov.sum() / cov.diagonal().sum()
  return sembl / ntraces

def tecva(X):
  fs = X.shape[1]
  rms = np.zeros(shape = X.shape)
  tecva = np.zeros(shape = X.shape)

  for i in range(X.shape[0]):
    signal = X[i, :]

    M, Minf, Msup = _sce(signal)
    rms[i, :] = _rms(signal, M, Minf, Msup)
    tecva[i, :] = np.imag(hilbert(rms[i, :]) * (-1))

  return tecva, rms

def _sce(signal, dx = 1):
    # dy = np.diff(signal) / dx

    M = 1
    # centro = np.argmax(dy)
    centro = np.argmax(np.abs(np.diff(np.angle(hilbert(signal)))))

    menor = signal[centro]
    parcial = menor
    k = 0
    while parcial <= menor:
      k -= 1
      menor = parcial
      try:
        parcial = signal[centro + k]
      except IndexError:
        break
    k += 1
    M += np.abs(k)
    Minf = np.abs(k)

    maior = signal[centro]
    parcial = maior
    k = 0
    while parcial >= maior:
      k += 1
      maior = parcial
      try:
        parcial = signal[centro + k]
      except IndexError:
        break
    k -= 1
    M += np.abs(k)
    Msup = np.abs(k)

    return M, Minf, Msup

def _rms(signal, M, Minf, Msup):
    rms = np.zeros(shape = signal.shape)
    for k in range(len(signal)):
      parcial = 0
      for kk in range(k - Minf, k + Msup + 1):
        try:
          parcial += signal[kk]**2
        except IndexError:
          pass
      rms[k] = np.sqrt(parcial / M)
    return rms

class Atributos():
  def __init__(self, path, resolucao, deltaz):
    self.path = path
    # self.X = np.load(self.path)
    self.X = segyio.tools.cube(path)[0, :, :]
    self.resolucao = resolucao

    self.extent = [1, self.X.shape[0] * self.resolucao, deltaz[0], deltaz[1]]

  def linhas(self, ariri, barravelha, camboriu):
    self.ariri = ariri
    self.barravelha = barravelha
    self.camboriu = camboriu

  def topo(self, marambaia):
    self.marambaia = marambaia

  def _complexo(self):
      self.envelope, self.fase, self.freq, self.sweetness = complexo(self.X)

  def _logaritmo(self):
    self.log = np.log(self.envelope)

  def _soterramento(self, marambaia = True):
    self.xx, self.yy = coordenadas(self.X)
    if marambaia:
      # extent = [8000, 2000]
      extent = self.entext
      linha = self.marambaia[:, 1]
      linha = (linha - extent[-1]) / (extent[-2] - extent[-1])
      self.soterramento = (self.yy.T + linha).T

  def _coherence(self):
    self.gersztenkorn, self.sobel, self.marfurt = coherence(self.X)

  def _tecva(self):
    self.tecva, self.rms = tecva(self.X)