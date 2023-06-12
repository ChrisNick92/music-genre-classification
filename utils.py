import os
import time

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import torch
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

CLASS_MAPPING = {'classical': 0, 'hiphop': 1, 'rock_metal_hardrock': 2, 'blues': 3}


def load_mfccs(mode):
    data_path = os.path.join("data", mode, "mfccs")
    X, y = np.load(os.path.join(data_path, "X.npy")).astype(np.float32), np.load(os.path.join(data_path, "labels.npy"))

    y = np.array([CLASS_MAPPING[label] for label in y], dtype=np.int64)

    return X, y


class VectorDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.tensor(y, dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def training_loop(epochs, model, train_dloader, val_dloader, optim, loss_fn, device='cpu', keep_best=False):

    train_loss, val_loss = 0., 0.
    _padding = len(str(epochs + 1))
    start = time.perf_counter()
    
    train_loss_list, val_loss_list = [], []

    for epoch in range(1, epochs + 1):

        model.train()
        with tqdm(train_dloader, unit='batch', leave=False, desc='Training set') as tbatch:
            for X, y in tbatch:
                # Forward pass
                X, y = X.to(device), y.to(device)
                out = model(X)
                loss = loss_fn(out, y)
                train_loss += loss.item()

                # Backward pass
                optim.zero_grad()
                loss.backward()
                optim.step()
            train_loss /= len(train_dloader)
            train_loss_list.append(train_loss)

        model.eval()
        y_true, y_pred = [], []
        best_f1 = -np.inf
        with torch.no_grad():
            with tqdm(val_dloader, unit='batch', leave=False, desc='Validation set') as vbatch:
                for X, y in vbatch:
                    X, y = X.to(device), y.to(device)
                    out = model(X)
                    loss = loss_fn(out, y)
                    preds = torch.argmax(out.detach().cpu(), dim=1)
                    y_true += y.detach().cpu().tolist()
                    y_pred += preds.tolist()
                    val_loss += loss.item()
            val_loss /= len(val_dloader)
            val_loss_list.append(val_loss)
            
            # Check for Best F1 score
            f1 = f1_score(y_true, y_pred, average='macro')
            if keep_best and f1 >= best_f1:
                best_epoch = epoch
                torch.save(model.state_dict(), "model.pt")
                best_f1 = f1
            

        print(f"Epoch {epoch:<{_padding}}/{epochs}. Train Loss: {train_loss:.3f}. Val Loss: {val_loss:.3f}")
        train_loss, val_loss = 0., 0.
    end = time.perf_counter()

    total = end - start
    print(f'\n- Total training time: {total:.2f} (secs)')
    if keep_best:
        print(f'- Best F1 score: {best_f1} on epoch {best_epoch}')
    
    fig, ax = plt.subplots(figsize = (8, 6))
    ax.plot(range(1, epochs+1), train_loss_list, label='Train Loss')
    ax.plot(range(1, epochs+1), val_loss_list, label='Val Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    plt.legend()
    plt.show()


def validation_loop(model, test_dloader, loss_fn):
    model.eval()
    total_loss = 0.
    y_true, y_pred = [], []
    with torch.no_grad():
        with tqdm(test_dloader) as tbatch:
            for X, y in tbatch:
                out = model(X)
                loss = loss_fn(out, y)
                total_loss += loss.item()

                batch_preds = torch.argmax(out, dim=1)
                y_true += y.tolist()
                y_pred += batch_preds.tolist()
        total_loss /= len(test_dloader)

    return total_loss, f1_score(y_true, y_pred,
                                average='macro'), accuracy_score(y_true, y_pred), confusion_matrix(y_true, y_pred)
