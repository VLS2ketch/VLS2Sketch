import numpy as np
from scipy.sparse import csr_matrix, load_npz
import argparse
import os

def read_data_from_file(filename):
    with open(filename, 'r') as file:
        data = file.read()
    return data

def write_array_to_file(array, filename):
    with open(filename, 'w') as f:
        for element in array:
            f.write(str(element) + ' ')

def save(csr_matrix, dataset, data_type, t):
    data_path = "data/" + dataset

    folder_path = data_path + f'/{data_type}'
    try:
        os.makedirs(folder_path)
        print("Successfully created a folder！")
    except FileExistsError:
        print("The folder already exists！")
    except Exception as e:
        print("An error occurred：", str(e))

    write_array_to_file(csr_matrix.data, f'{folder_path}/values_' + str(t) + '.txt')
    write_array_to_file(csr_matrix.indices, f'{folder_path}/indices_' + str(t) + '.txt')
    write_array_to_file(csr_matrix.indptr, f'{folder_path}/indptr_' + str(t) + '.txt')

    # Save rows, cols, and nnz to info.txt
    with open(f'{folder_path}/info_' + str(t) + '.txt', 'w') as f:
        f.write(f"{csr_matrix.shape[0]} {csr_matrix.shape[1]} {csr_matrix.nnz}")


def preProcess(G_cur, G_next, A_cur, A_next, t):
    nodes_new = G_next.shape[0] - G_cur.shape[0]
    attrs_new = A_next.shape[1] - A_cur.shape[1]
    print(f'node_new:{nodes_new}---attrs_new:{attrs_new}')

    # A node is added. G inserts all zeros in row and column node_new at the bottom and rightmost, and A inserts all zeros in row and column node_new at the bottom
    if nodes_new > 0:
        print('Added node')
        G_cur.resize((G_next.shape[0], G_next.shape[0]))
        A_cur.resize((G_next.shape[0], A_cur.shape[1]))

    # New attributes have been added, and A inserts an all-zero element in the attrs_new column on the far right
    if attrs_new > 0:
        print('Added properties')
        A_cur.resize((A_next.shape[0],A_next.shape[1]))

    decay = 1 / np.e
    # Multiply all the elements of the sparse matrix by 1/e
    A_cur_decay = A_cur.multiply(decay)

    delta_A = A_next - A_cur_decay
    delta_G = G_next - G_cur

    save(G_cur, dataset, 'G_cur', t)
    save(delta_G, dataset, 'delta_G', t)
    save(delta_A, dataset, 'delta_A', t)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process...')
    parser.add_argument('--dataset', type=str, help='graph dataset name')
    parser.add_argument('--timestep', type=int, help='the time step of dynamic data')
    parser.add_argument('--start_year', type=int, help='start year of dynamic data')

    args = parser.parse_args()
    dataset = args.dataset
    timestep = args.timestep
    start_year = args.start_year

    data_path = "data/" + dataset

    adjOld_dir = data_path + '/network_' + str(start_year) + '.npz'
    attrsOld_dir = data_path + '/attrs_' + str(start_year) + '.npz'
    adjacency_old = load_npz(adjOld_dir)
    attributes_old = load_npz(attrsOld_dir)
    attributes_old = csr_matrix(attributes_old)
    adjacency_old = csr_matrix(adjacency_old)

    save(attributes_old, dataset, 'attrs', start_year)
    save(adjacency_old, dataset, 'network', start_year)

    print(f'adj:{adjacency_old.shape}--attrs:{attributes_old.shape}')
    print(f'adj_nnz:{adjacency_old.nnz}--attrs_nnz:{attributes_old.nnz}')


    for t in range(start_year + 1, start_year + timestep):
        adj_dir = data_path + '/network_' + str(t) + '.npz'
        attrs_dir = data_path + '/attrs_' + str(t) + '.npz'
        adjacency = load_npz(adj_dir)
        attributes = load_npz(attrs_dir)
        adjacency = csr_matrix(adjacency)
        attributes = csr_matrix(attributes)
        print(f'adj:{adjacency.shape}--attrs:{attributes.shape}')
        print(f'adj_nnz:{adjacency.nnz}--attrs_nnz:{attributes.nnz}')

        # Save rows, cols, and nnz to info.txt
        with open(f'{data_path}/attrs/info_' + str(t) + '.txt', 'w') as f:
            f.write(f"{attributes.shape[0]} {attributes.shape[1]} {attributes.nnz}")

        preProcess(adjacency_old, adjacency, attributes_old, attributes, t)
        adjacency_old = adjacency
        attributes_old = attributes_old



