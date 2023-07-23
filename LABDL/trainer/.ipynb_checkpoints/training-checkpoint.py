import torch 
import torch.nn as nn
import torch.nn.functional as F 

from .utils import plot 

import wandb 

def train(model: nn.Module, 
         dataloader_train: torch.utils.data.DataLoader, 
         regression: bool = True): 
    """
    train model with Adam optimizer on given dataset.
    """    
    lr = 1e-2
    num_epoches = 500
    log_frequency = 100
    
    #run = wandb.init(
    #    project="LABDL_vanilla", 
    #    config={
    #        "learning_rate": lr, 
    #        "epochs": num_epoches
    #    })
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss() if regression else nn.BCELoss()
    
    losses_train = []
    for epoch in range(num_epoches):
            # train for 10 epochs
            # forward + backward + optimize 
        
            loss_train = train_epoch(criterion, model, dataloader_train, optimizer)
            
            #wandb.log({
            #    "train_loss": loss_train, 
            #    "val_loss": loss_val, 
            #    })
            
            # save statistics
            losses_train.append(loss_train.detach().cpu().numpy().item())
            #losses_val.append(loss_val.detach().cpu().numpy().item())
            
            # print statistics
            if epoch % log_frequency == 0: 
                print(f"=============EPOCH {epoch+1}==============")
                print(
                    "loss_train: %.3f"
                    % (loss_train.detach().cpu().numpy().item()))
                #fig = plot(dataset, model)
                #wandb.log({
                #    "fit": fig
                #})
    print("Finished Training.")
    loss = sum(losses_train)
    return loss
        

def train_epoch(criterion, 
                model: nn.Module, 
                dataloader: torch.utils.data.DataLoader, 
                optimizer: torch.optim.Adam):
    model.train()
    avg_loss = 0.0
    for batch in dataloader:
        features, targets = batch

        optimizer.zero_grad()

        preds = model(features)

        step_loss = criterion(preds, targets)

        step_loss.backward()
        optimizer.step()

        avg_loss += step_loss
        
    return avg_loss / len(dataloader)
    
def evaluate_epoch(criterion, 
                   model: nn.Module, 
                   dataloader: torch.utils.data.DataLoader, 
                   optimizer: torch.optim.Adam):
    model.eval()
    avg_loss = 0.0
        
    for batch in dataloader:
        # Accumulates loss in dataset. 
        with torch.no_grad():
            features, targets = batch

            preds = model(features)

            step_loss = criterion(preds, targets)
                
            avg_loss += step_loss

    return avg_loss / len(dataloader)
    
