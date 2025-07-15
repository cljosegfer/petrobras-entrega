
import pandas as pd
import numpy as np

class Superficie():
  def __init__(self, path, offsets = [2100, 3650], shape = [2898, 1345]):
    self.path = path
    self.id = path.split('/')[-1].split('.')[0]
    columns = ['inline', 'crossline', 'z']
    self.data = pd.read_csv(self.path, header = None, names = columns, comment = '#')
    
    # self.offsets = [2100, 3650]
    # self.shape = [2898, 1345]
    self.offsets = offsets
    self.shape = shape

  def line(self, i, inline = 0, offset = False):
    if inline == 0:
      linha = self.data.loc[self.data['inline'] == i + offset * self.offsets[1]][['z', 'crossline']].values
    else:
      linha = self.data.loc[self.data['crossline'] == i + offset * self.offsets[0]][['z', 'inline']].values
    linha = self._remove_offset(linha, inline)
    linha = self._interpolate(linha)
    return linha

  def _remove_offset(self, linha, inline = 0):
    linha[:, -1] -= self.offsets[inline]
    lim = self.shape[inline] - 1
    borda = (linha[:, -1] > lim) + (linha[:, -1] < 0)
    linha = linha[~borda, :]
    return linha

  def _interpolate(self, linha):
    i = 0
    log = []
    for ii, coluna in enumerate(linha):
      if coluna[-1] == i:
        log.append(coluna)
        i += 1
      else:
        delta = coluna[-1] - i
        xp = np.vstack([linha[ii-1], coluna])
        for j in range(int(delta)):
          interpolado = np.interp(i + j, xp[:, 1], xp[:, 0])
          log.append([interpolado, i + j])
        log.append(coluna)
        i += j + 2
    log = np.array(log)

    for coluna in linha:
      assert (coluna == log[int(coluna[-1]), :]).all()
    linha = np.flip(log, axis = 1)
    return linha