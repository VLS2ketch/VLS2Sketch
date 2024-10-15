import numpy as np
import argparse

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

def write_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for element in array:
            f.write(str(element) + ' ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    parser.add_argument('--K', type=int, help='embedding dimensionality')
    parser.add_argument('--R', type=int, help='number of iterations')
    parser.add_argument('--cur_year', type=int, help='current year of dynamic data')

    args = parser.parse_args()
    dataset = args.dataset
    R = args.R
    K = args.K
    cur_year = args.cur_year
    data_path = "data/" + dataset + '/delta_G'
    info_path = data_path + '/info_' + str(cur_year) + '.txt'
    info = read_data_from_file(info_path)
    info = [int(i) for i in info.split()]
    n_delta, m_delta, nnz_delta = info

    for r in range(R + 1):
        indptr_path = 'X_list/' + dataset + '/indptr_R' + str(r) + '_K' + str(K) + '.txt'
        info_path = 'X_list/' + dataset + '/info_R' + str(r) + '_K' + str(K) + '.txt'

        info = read_data_from_file(info_path)
        info = [int(i) for i in info.split()]
        n, m, nnz = info
        nodes_new = n_delta - n

        if(nodes_new>0):

            print("process indptr")
            indptr = read_data_from_file(indptr_path)
            indptr = [int(i) for i in indptr.split()]

            new_indptr = np.full(nodes_new, indptr[n])
            concatenated_array = np.concatenate((indptr, new_indptr))

            print("n:", n)
            print("m:", m)
            print("nnz:", nnz)
            print("saving......")

            # Save rows, cols, and nnz to info.txt
            with open(info_path, 'w') as f:
                f.write(f"{n_delta} {m} {nnz}")
            write_array_to_file(concatenated_array, indptr_path)



