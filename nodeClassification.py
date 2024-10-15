import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score,accuracy_score
from sklearn.model_selection import KFold
import random
import argparse

# multi-class classification
def predict_cv(X, y, C=1., num_workers=1):
    micro, macro, accuracy = [], [], []
    kf = KFold(n_splits=10, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = OneVsRestClassifier(
                LogisticRegression(
                    C=C,
                    solver="liblinear",
                    multi_class="ovr"),
                n_jobs=num_workers)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        mi = f1_score(y_test, y_pred, average="micro")
        ma = f1_score(y_test, y_pred, average="macro")
        micro.append(mi)
        macro.append(ma)
        accuracy.append(acc)
    print(f'acc: {np.mean(accuracy):.4f}----micro: {np.mean(micro):.4f}----macro: {np.mean(macro):.4f}')
    return accuracy, micro, macro

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
      print(f'R={R}')
      for t in range(start_year, start_year + timestep):
          print(f't={t}')
          labels_path = 'data/' + dataset + '/labels_' + str(t) + '.npy'
          labels = np.load(labels_path)

          accuracies = []
          micro_f1_scores = []
          macro_f1_scores = []

          embedding_path = 'results/' + dataset + '/embeddings/embedding.R' + str(R) + '.time' + str(t) +   '_K' + str(K) + '.npz'
          embedding = load_npz(embedding_path)
          embedding = csr_matrix(embedding)
          embedding = hstack([embedding, csr_matrix(np.logical_not(embedding.toarray()))])
          embedding = csr_matrix(embedding)
          embedding = embedding>0

          for iteration in range(1,6):
              subset_size = 60000  # Set the subset size as needed
              if dataset == 'ogbn-papers100M':
                  non_negative_indices = np.where(labels != -1)[0]
                  random_indices = random.sample(list(non_negative_indices), subset_size)
                  random_indices = np.array(random_indices)
              else:
                  random_indices = np.random.choice(embedding.shape[0], subset_size, replace=False)
              sub_embedding = embedding[random_indices]
              sub_labels = labels[random_indices]
              acc,micro,macro = predict_cv(sub_embedding,sub_labels)

              accuracies.extend(acc)
              micro_f1_scores.extend(micro)
              macro_f1_scores.extend(macro)

          average_accuracy = np.mean(accuracies)
          average_micro_f1 = np.mean(micro_f1_scores)
          average_macro_f1 = np.mean(macro_f1_scores)

          std_accuracy = np.std(accuracies)
          std_micro_f1 = np.std(micro_f1_scores)
          std_macro_f1 = np.std(macro_f1_scores)
          print(f'【R={R},K={K}】acc：{average_accuracy * 100:.4f}--std_accuracy：{std_accuracy * 100:.4f}--micro-f1: {average_micro_f1 * 100:.4f}--std_micro：{std_micro_f1 * 100:.4f}--macro-f1: {average_macro_f1 * 100:.4f}--std_macro：{std_macro_f1 * 100:.4f}')

          log_path = '../results/' + dataset + '/nc/multi-class' + str(t) + '.log'
          with open(log_path, "a") as fout:
              fout.write("data: "+dataset+"\n")
              fout.write("dimensionality: "+str(K)+"\n")
              fout.write("iteration: "+str(R)+"\n")
              fout.write("timestep: " + str(t) + "\n")
              fout.write("accuracy: "+str(average_accuracy)+"---std_accuracy: "+str(std_accuracy)+"\n")
              fout.write("micro-f1: "+str(average_micro_f1)+"---std_micro-f1: "+str(std_micro_f1)+"\n")
              fout.write("macro-f1: "+str(average_macro_f1)+"---std_macro-f1: "+str(std_macro_f1)+"\n")
              fout.write("----------------------------------------------------------------------------\n")



