from scipy.sparse import csr_matrix, save_npz
import os
import argparse

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

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

    for R in range(1,6):
        for t in range(start_year, start_year + timestep):
            print(f't:{t}')
            data_path = 'results/' + dataset + '/emb_cpp/values_R' + str(R) + '_timestep' + str(t) +  '_K' + str(K) + '.txt'
            indices_path = 'results/' + dataset + '/emb_cpp/indices_R' + str(R) + '_timestep' + str(t) +  '_K' + str(K) + '.txt'
            indptr_path = 'results/' + dataset + '/emb_cpp/indptr_R' + str(R) + '_timestep' + str(t) +  '_K' + str(K) + '.txt'
            info_path = 'results/' + dataset + '/emb_cpp/info_R' + str(R) + '_timestep' + str(t) +  '_K' + str(K) + '.txt'
            print("process data")

            data = read_data_from_file(data_path)
            data = [int(i)>0 for i in data.split()]

            print("process indptr")
            indptr = read_data_from_file(indptr_path)
            indptr = [int(i) for i in indptr.split()]
            print("process indices")

            indices = read_data_from_file(indices_path)
            indices = [int(i) for i in indices.split()]

            info = read_data_from_file(info_path)
            info = [int(i) for i in info.split()]
            n, m, nnz = info

            print("n:", n)
            print("m:", m)
            print("nnz:", nnz)
            print("creating......")

            sparse_matrix = csr_matrix((data, indices, indptr), shape=(n, m))

            print('save.....')

            folder_path = 'results/' + f'{dataset}' + '/embeddings'
            try:
                os.makedirs(folder_path)
                print("Successfully created a folder！")
            except FileExistsError:
                print("The folder already exists！")
            except Exception as e:
                print("An error occurred：", str(e))

            f_path = folder_path +'/embedding.R' + str(R) +'.time' + str(t) +   '_K' + str(K) + '.npz'
            save_npz(f_path,sparse_matrix)