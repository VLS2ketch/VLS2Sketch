import numpy as np
from scipy.sparse import random
import argparse
import os

def generate_sparse_matrix(m, K):
    density = 1 / np.sqrt(m)
    sparse_matrix = random(m, K, density=density, format='csr')
    sparse_matrix.data = sparse_matrix.data * 2 - 1
    return sparse_matrix

def write_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for element in array:
            f.write(str(element) + ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--timestep', type=int, help='the time step of dynamic data')
    parser.add_argument('--start_year', type=int, help='start year of dynamic data')

    args = parser.parse_args()
    dataset = args.dataset
    K = args.K
    timestep = args.timestep
    start_year = args.start_year

    print(f'--------------------------{dataset} Datasets-----------------------------------------------------')
    print(f'the embedding dimensionality is K = {K}')

    for t in range(start_year, start_year + timestep):

      attrs_path = 'data/' + dataset + '/attrs/info_' + str(t) + '.txt'
      with open(attrs_path, 'r') as file:
          data = file.read()
      data = [int(i) for i in data.split()]
      n, m, nnz = data
      R = generate_sparse_matrix(m,K)

      data_path = "data/" + dataset
      folder_path = data_path + f'/SRMatrix'
      try:
          os.makedirs(folder_path)
          print("Successfully created a folder！")
      except FileExistsError:
          print("The folder already exists！")
      except Exception as e:
          print("An error occurred：", str(e))

      write_array_to_file(R.data, f'{folder_path}/values_' + str(t) + '_K' + str(K) + '.txt')
      write_array_to_file(R.indices, f'{folder_path}/indices_' + str(t) + '_K' + str(K) + '.txt')
      write_array_to_file(R.indptr, f'{folder_path}/indptr_' + str(t) + '_K' + str(K)+ '.txt')

      # Save rows, cols, and nnz to info.txt
      with open(f'{folder_path}/info_' + str(t) + '_K' + str(K) + '.txt', 'w') as f:
          f.write(f"{R.shape[0]} {R.shape[1]} {R.nnz}")

