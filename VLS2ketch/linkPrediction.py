import numpy as np
from scipy.sparse import load_npz
import argparse


def linkPrediction(dataset,R,K,timestep,start_year):
    ratio = 0.9
    np.random.seed(42)  # Set a seed for reproducibility

    for t in range(start_year, start_year + timestep):
        file_name = 'data/' + dataset + '/network_' + str(t) + '.npz'
        network = load_npz(file_name)

        nodeNum = network.shape[0]
        network.setdiag(0)

        auc_scores = []

        embedding_path = '../results_T/' + dataset + '/embeddings/embedding.R' + str(R) + '.time' + str(t) +   '_K' + str(K) + '.npz'
        embedding = load_npz(embedding_path)
        embedding.setdiag(0)
        embedding = embedding.astype(int)
        embedding = embedding.toarray()

        file_name2 = "data/" + dataset + "/linkPrediction/" + dataset + "_trainGraph_t" + str(t)  + "_" + str(ratio) + ".npz"
        file_name3 = "data/" + dataset + "/linkPrediction/" + dataset + "_testGraph_t" + str(t)  + "_" + str(ratio) + ".npz"
        trainGraph = load_npz(file_name2)
        testGraph = load_npz(file_name3)

        for i in range(5):
            count = 1
            times = 10000
            nonexistence = np.zeros((2, times), dtype=int)
            np.random.seed(0)
            while count <= times:
                edgeIds = np.random.randint(0, nodeNum, size=(2, 1))
                if network[edgeIds[0], edgeIds[1]] == 0:
                    nonexistence[:, count - 1] = edgeIds.flatten()
                    count += 1

            trainGraph.setdiag(0)
            nonexistence_similarity = np.sum(
                embedding[nonexistence[0], :] == embedding[nonexistence[1], :], axis=1
            )

            testGraph.setdiag(0)
            i_test, j_test = testGraph.nonzero()

            testedEdges = np.column_stack((i_test, j_test))
            testedEdges = testedEdges[testedEdges[:, 0] > testedEdges[:, 1]]

            selected_edges = np.random.choice(testedEdges.shape[0], times, replace=True)
            testedEdges = testedEdges[selected_edges, :]

            missing_similarity = np.sum(
                embedding[testedEdges[:, 0], :] == embedding[testedEdges[:, 1], :], axis=1
            )

            greatNum = np.sum(missing_similarity > nonexistence_similarity)
            equalNum = np.sum(missing_similarity == nonexistence_similarity)

            auc = (greatNum + 0.5 * equalNum) / times
            auc_scores.append(auc)


        average_auc = np.mean(auc_scores)
        std_auc = np.std(auc_scores)

        print(f't:{t}---auc:{average_auc * 100}---std_auc:{std_auc * 100}')

        log_path = '../results/' + dataset + '/lp/linkPrediction' + str(t) + '.log'
        with open(log_path, "a") as fout:
            fout.write("data: " + dataset + "\n")
            fout.write("dimensionality: " + str(K) + "\n")
            fout.write("iteration: " + str(R) + "\n")
            fout.write(f"auc: {average_auc}\n")
            fout.write("----------------------------------------------------------------------------\n")

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
        linkPrediction(dataset,R,K,timestep,start_year)
