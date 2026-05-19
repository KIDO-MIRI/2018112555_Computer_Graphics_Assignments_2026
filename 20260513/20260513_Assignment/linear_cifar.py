import numpy as np
import pickle
import tarfile
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

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

mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image

class LinearNet(nn.Module):
    def __init__(self):
        super(LinearNet, self).__init__()
        self.linear = nn.Linear(3072, 10) [cite: 6]

    def forward(self, x):
        return self.linear(x)

class LinearClassifier:
    def __init__(self):
        self.model = LinearNet()
        self.criterion = nn.CrossEntropyLoss()

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=1000, batch_size=200):
        num_train = X.shape[0]
        
        optimizer = optim.SGD(self.model.parameters(), lr=learning_rate, weight_decay=reg)
        
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y).long()

        loss_history = []
        self.model.train()
        
        for it in range(num_iters):

            batch_indices = np.random.choice(num_train, batch_size, replace=True)
            X_batch = X_tensor[batch_indices]
            y_batch = y_tensor[batch_indices]

            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss_history.append(loss.item())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        return loss_history

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.from_numpy(X)
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, y_pred = torch.max(outputs, 1)
        return y_pred.numpy()

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

print("\n[Start 5-Fold Cross-Validation for Hyperparameters]")
num_folds = 5

learning_rates = [1e-3, 1e-2, 5e-2]
regularization_strengths = [1e-4, 1e-3, 1e-2]

X_train_folds = np.array_split(x_train, num_folds)
y_train_folds = np.array_split(y_train, num_folds)

results = {}
best_val = -1
best_lr = -1
best_reg = -1

for lr in learning_rates:
    for reg in regularization_strengths:
        val_accuracies = []
        print(f"Testing lr={lr:.1e}, reg={reg:.1e} with 5 folds...")
        
        for i in range(num_folds):
            X_val_fold = X_train_folds[i]
            y_val_fold = y_train_folds[i]
            
            X_train_fold = np.concatenate([X_train_folds[j] for j in range(num_folds) if j != i])
            y_train_fold = np.concatenate([y_train_folds[j] for j in range(num_folds) if j != i])
            
            classifier = LinearClassifier()
            classifier.train(X_train_fold, y_train_fold, learning_rate=lr, reg=reg, num_iters=300, batch_size=200)
            
            y_val_pred = classifier.predict(X_val_fold)
            val_accuracies.append(np.mean(y_val_pred == y_val_fold))
        
        mean_val_acc = np.mean(val_accuracies)
        results[(lr, reg)] = mean_val_acc
        
        if mean_val_acc > best_val:
            best_val = mean_val_acc
            best_lr = lr
            best_reg = reg

print("\n[Cross-Validation Results]")
for (lr, reg), val_acc in results.items():
    print(f"lr={lr:.1e}, reg={reg:.1e} | Mean Val Acc: {val_acc*100:.2f}%")

print(f"\nBest Validation Accuracy: {best_val*100:.2f}% (lr={best_lr:.1e}, reg={best_reg:.1e})")

print("\n[Training Best Model on Full Training Set]")
best_classifier = LinearClassifier()
loss_hist = best_classifier.train(x_train, y_train, learning_rate=best_lr, reg=best_reg, num_iters=1500, batch_size=200)

plt.figure(figsize=(10, 6))
plt.plot(loss_hist, label='Training Loss')
plt.title('Loss History during Training (Best Model)')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("\n[Start Final Evaluation on Test Set]")
pred_test = best_classifier.predict(x_test)
test_acc = np.mean(pred_test == y_test)
print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
print_confusion_matrix(y_test, pred_test)