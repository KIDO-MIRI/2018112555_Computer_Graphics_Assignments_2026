import numpy as np
import pickle
import tarfile
import os
import matplotlib.pyplot as plt

def load_cifar10_dataset(tar_filepath):
    extract_dir = 'cifar-10-batches-py'
    
    if not os.path.exists(extract_dir):
        print("Unzip Data...")
        with tarfile.open(tar_filepath, 'r:gz') as tar:
            tar.extractall()
            
    def load_pickle_batch(filename):
        with open(filename, 'rb') as f:
            datadict = pickle.load(f, encoding='bytes')
            X = datadict[b'data']
            Y = datadict[b'labels']
            return X, Y

    print("Read Dataset...")
    X_train_list, Y_train_list = [], []
    for i in range(1, 6):
        batch_path = os.path.join(extract_dir, f'data_batch_{i}')
        X, Y = load_pickle_batch(batch_path)
        X_train_list.append(X)
        Y_train_list.append(Y)
        
    x_train = np.concatenate(X_train_list).astype(np.float32)
    y_train = np.concatenate(Y_train_list)
    
    test_batch_path = os.path.join(extract_dir, 'test_batch')
    x_test, y_test = load_pickle_batch(test_batch_path)
    x_test = np.array(x_test).astype(np.float32)
    y_test = np.array(y_test)
    
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_cifar10_dataset('cifar-10-python.tar.gz')

num_training = 5000
num_test = 1000

x_train = x_train[:num_training]
y_train = y_train[:num_training]
x_test = x_test[:num_test]
y_test = y_test[:num_test]

class KNearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, metric='L2'):
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))

        if metric == 'L2':
            dists = np.sqrt(
                np.sum(X**2, axis=1, keepdims=True) 
                - 2 * np.dot(X, self.X_train.T) 
                + np.sum(self.X_train**2, axis=1)
            )
        elif metric == 'L1':
            for i in range(num_test):
                dists[i, :] = np.sum(np.abs(self.X_train - X[i, :]), axis=1)
                
        return self.predict_labels(dists, k=k)

    def predict_labels(self, dists, k=1):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        
        for i in range(num_test):
            closest_y = self.y_train[np.argsort(dists[i, :])[:k]]
            y_pred[i] = np.bincount(closest_y).argmax()
            
        return y_pred.astype(int)

def print_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
        
    print("\n[Confusion Matrix]")
    print("Pred:  0  1  2  3  4  5  6  7  8  9")
    print("-" * 37)
    for i in range(num_classes):
        row_str = f"T{i}  | " + " ".join([f"{val:2d}" for val in cm[i]])
        print(row_str)
    print("\n")

print("\n[Start 5-Fold Cross-Validation]")
num_folds = 5
k_choices = [1, 3, 5, 7, 9]

X_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

k_to_accuracies_l1 = {}
k_to_accuracies_l2 = {}
cv_classifier = KNearestNeighbor()

for k in k_choices:
    k_to_accuracies_l1[k] = []
    k_to_accuracies_l2[k] = []
    
    print(f"Testing K={k} with 5 folds...")
    for i in range(num_folds):
        X_val_fold = X_train_folds[i]
        y_val_fold = y_train_folds[i]
        
        X_train_fold = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
        y_train_fold = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
        
        cv_classifier.train(X_train_fold, y_train_fold)
        
        y_val_pred_l1 = cv_classifier.predict(X_val_fold, k=k, metric='L1')
        k_to_accuracies_l1[k].append(np.mean(y_val_pred_l1 == y_val_fold))
        
        y_val_pred_l2 = cv_classifier.predict(X_val_fold, k=k, metric='L2')
        k_to_accuracies_l2[k].append(np.mean(y_val_pred_l2 == y_val_fold))

mean_acc_l1 = [np.mean(k_to_accuracies_l1[k]) for k in k_choices]
mean_acc_l2 = [np.mean(k_to_accuracies_l2[k]) for k in k_choices]

print("\n[Cross-Validation Results]")
for i, k in enumerate(k_choices):
    print(f"K={k} | Mean L1 Acc: {mean_acc_l1[i]*100:.2f}% | Mean L2 Acc: {mean_acc_l2[i]*100:.2f}%")

plt.figure(figsize=(10, 6))
plt.plot(k_choices, mean_acc_l1, marker='o', label='L1 Distance')
plt.plot(k_choices, mean_acc_l2, marker='o', label='L2 Distance')
plt.title('5-Fold Cross-Validation on k')
plt.xlabel('K value')
plt.ylabel('Mean cross-validation accuracy')
plt.xticks(k_choices)
plt.legend()
plt.grid(True)
plt.show()

print("\n[Start Final Evaluation on Test Set]")
final_classifier = KNearestNeighbor()
final_classifier.train(x_train, y_train)

best_k = k_choices[np.argmax(mean_acc_l1)]
print(f"Best K from CV: {best_k}. Reporting final results on test set.")

for k in k_choices:
    print(f"======== Final Test K = {k} ========")
    
    pred_l1 = final_classifier.predict(x_test, k=k, metric='L1')
    acc_l1 = np.mean(pred_l1 == y_test)
    print(f"K={k} / L1 Accuracy: {acc_l1 * 100:.2f}%")
    print_confusion_matrix(y_test, pred_l1)
    
    pred_l2 = final_classifier.predict(x_test, k=k, metric='L2')
    acc_l2 = np.mean(pred_l2 == y_test)
    print(f"K={k} / L2 Accuracy: {acc_l2 * 100:.2f}%")
    print_confusion_matrix(y_test, pred_l2)