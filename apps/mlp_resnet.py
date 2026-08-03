import sys

sys.path.append("../python")
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    main_path = nn.Sequential(
        nn.Linear(dim,hidden_dim),
        norm,
        nn.Relu(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim,dim),
        norm
    )
    return nn.Sequential(
        nn.Residual(main_path),
        nn.Relu()
    )


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    modules = [nn.Linear(dim,hidden_dim),nn.Relu()]
    for _ in range(num_blocks):
        modules.append(ResidualBlock(dim,hidden_dim,norm,drop_prob))
    modules.append(nn.Linear(hidden_dim,num_classes))
    return nn.Sequential(*modules)


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    loss_fn = nn.SoftmaxLoss()
    if opt is not None:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_error = 0
    total_samples = 0

    for X,y in dataloader:
        logits = model(X)
        loss = loss_fn(logits,y)

        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()

        batch_size = X.shape[0]
        total_samples += batch_size
        total_loss += loss.numpy().item() * batch_size

        preds = np.argmax(logits.numpy(), axis=1)
        total_error += np.sum(preds != y.numpy())

    avg_loss = total_loss / total_samples
    avg_error = total_error / total_samples
    return avg_error, avg_loss



def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    raise NotImplementedError()
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")
