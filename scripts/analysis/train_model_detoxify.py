from early_stopping_pytorch.pytorchtools import EarlyStopping
from degree import degree_pna
import argparse
from torch_geometric.loader import DataLoader
from torch_geometric.loader.imbalanced_sampler import ImbalancedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, brier_score_loss
import torch
import numpy as np
from models import BinaryClass
from torchmetrics.wrappers import BootStrapper
import matplotlib.pyplot as plt
import warnings
from torchmetrics.classification import BinaryAUROC, PrecisionRecallCurve

warnings.filterwarnings("ignore")
torch.manual_seed(12345)
np.random.seed(1234)

def main(graphs):
    dataset = graphs
    #np.random.shuffle(dataset) 
    
    split = int(len(dataset)*0.7)
    train_dataset = dataset[:split]

    remaining_dataset = dataset[split:]
    remaining_split = int(len(remaining_dataset)*0.5)

    validation_dataset = remaining_dataset[:remaining_split]
    test_dataset = remaining_dataset[remaining_split:]
    print(f'Number of training graphs: {len(train_dataset)}')
    print(f'Number of validation graphs: {len(validation_dataset)}')
    print(f'Number of test graphs: {len(test_dataset)}')
    # Define the data loaders
    train_loader = DataLoader(train_dataset,sampler=ImbalancedSampler(train_dataset), batch_size=64)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)
    deg = degree_pna(train_dataset)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BinaryClass(deg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                                  min_lr=0.00001)
    m = torch.nn.Sigmoid()
    loss_fn = torch.nn.BCEWithLogitsLoss()

    early_stopping = EarlyStopping(patience=20)

    history = {
        'train_loss': [],
        'val_loss': []
    }

    def train():
      model.train()
      total_loss = 0
      for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        loss = loss_fn(out.squeeze(), data.y.to(torch.float32))
        loss.backward()
        total_loss += loss.item() * data.num_graphs
        optimizer.step()
      return total_loss / len(train_loader.dataset)


    @torch.no_grad()
    def val(loader):
        model.eval()
        total_error = 0
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            total_error += loss_fn(out.squeeze(), data.y.to(torch.float32)).item() * data.num_graphs
        return total_error / len(loader.dataset)

    @torch.no_grad()
    def test(model_in, loader):
        model_in.eval()
        total_error = 0
        y_true = []
        y_score = []
        for data in loader:
            data = data.to(device)
            out = model_in(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = m(out.squeeze()) 
            total_error += loss_fn(out.squeeze(), data.y.to(torch.float32)).item() * data.num_graphs
            y_true.extend(data.y.tolist())
            y_score.extend(pred.tolist())
             
        y_true = np.array(y_true)
        y_score = np.array(y_score)
        
        pr_base = (y_true == 1).sum() / len(y_true)
        acc = (torch.tensor(y_true > 0.5) == torch.tensor(y_score > 0.5)).sum() / len(y_score)
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        pr_auc_score = auc(recall, precision)
        roc_score = roc_auc_score(y_true, y_score)
        brier_score = brier_score_loss(y_true, y_score)
        return total_error / len(loader.dataset), roc_score, acc, brier_score, pr_auc_score, pr_base

    for epoch in range(1, 100):
        loss = train()
        val_loss = val(validation_loader)
        history['train_loss'].append(loss)
        history['val_loss'].append(val_loss)

        test_mae, roc, acc, brier, pr_auc, pr_base = test(model, test_loader)
        
        if epoch >= 5:
            early_stopping(val_loss, model, epoch)
        
        scheduler.step(val_loss)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_loss:.4f}, '
              f'Test: {test_mae:.4f}, ROC: {roc:.4f}, ACC: {acc:.4f}, '
              f'PR_AUC: {pr_auc:.4f}, PR_BASE: {pr_base:.4f} BRIER: {brier:.4f} ')
        if early_stopping.early_stop:
            print("Early stopping")
            break
    checkpoint_state = torch.load("checkpoint.pt")
    model.load_state_dict(checkpoint_state)
    print("FINAL RESULTS: \n\n")
    test_mae, roc, acc, brier, pr_auc, pr_base= test(model, test_loader)
    print(f'Test: {test_mae:.4f}, ROC: {roc:.4f}, ACC: {acc:.4f}, '
              f'PR_AUC: {pr_auc:.4f}, PR_BASE: {pr_base:.4f} BRIER: {brier:.4f} ')
    # Plotting
    plt.axvline(x=early_stopping.early_stop_epoch, color='r', label=f'Early Stop')
    plt.plot(range(1,len(history['train_loss'])+1), history['train_loss'], label="Train Loss")
    plt.plot(range(1,len(history['val_loss'])+1),history['val_loss'], label="Val Loss")
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-d", "--dataset", help="Path to dataset", type=str)
    args = arg_parser.parse_args()
    graphs = torch.load(args.dataset)
    main(graphs)