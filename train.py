import argparse
import datetime
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import build_from_config

#import warnings
#warnings.filterwarnings('ignore')

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that takes dataset path, model saving path, and tensorboard log path as arguments.")

    # Add arguments
    parser.add_argument('--model_saving_path', required=True, help='Path to save the model')
    parser.add_argument('--tensorboard_log_path', required=True, help='Path to save TensorBoard logs')
    parser.add_argument("--config", "-c", required=True, type=str, help="Path to the config file.")

    return parser.parse_args()   

def train(epoch, train_dataloader, val_dataloader, writer):
    model.train()
    train_losses = []
    train_accuracies = []
    for iterations, (images, labels) in enumerate(train_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        output = F.softmax(output, dim = 1)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        pred = torch.argmax(output, dim=1)
        acc = (pred == labels).sum() / len(labels)
        #import pdb; pdb.set_trace()
        train_accuracies.append(acc)
        
        if(iterations % 100 == 0):
            print(f"Batch [{iterations}/{len(train_dataloader)}] : Training Loss {sum(train_losses) / len(train_losses)}, Training Acc {sum(train_accuracies) / len(train_accuracies)}")
    
    model.eval()
    val_losses = []
    val_accuracies = []
    for iterations, (images, labels) in enumerate(val_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        output = F.softmax(output, dim = 1)
        loss = criterion(output, labels)
        
        val_losses.append(loss.item())
        
        pred = torch.argmax(output, dim=1)
        acc = (pred == labels).sum() / len(labels)
        val_accuracies.append(acc)
        
    train_losses = torch.tensor(train_losses, dtype=torch.float32)
    train_accuracies = torch.tensor(train_accuracies, dtype=torch.float32)
    val_losses = torch.tensor(val_losses, dtype=torch.float32)
    val_accuracies = torch.tensor(val_accuracies, dtype=torch.float32)
    writer.add_scalar("Loss/train", train_losses.mean(), epoch)
    writer.add_scalar("Accuracy/train", train_accuracies.mean(), epoch)
    writer.add_scalar("Loss/val", val_losses.mean(), epoch)
    writer.add_scalar("Accuracy/val", val_accuracies.mean(), epoch)
    
    scheduler.step()
    return train_losses.mean(), train_accuracies.mean(), val_losses.mean(), val_accuracies.mean()

def test(epoch, test_dataloader, writer):
    model.eval()
    test_losses = []
    test_accuracies = []
    for iterations, (images, labels) in enumerate(test_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        output = F.softmax(output, dim = 1)
        loss = criterion(output, labels)
        
        test_losses.append(loss.item())
        
        pred = torch.argmax(output, dim=1)
        acc = (pred == labels).sum() / len(labels)
        test_accuracies.append(acc)
    
    test_losses = torch.tensor(test_losses, dtype=torch.float32)
    test_accuracies = torch.tensor(test_accuracies, dtype=torch.float32)
    writer.add_scalar("Loss/test",test_losses.mean(), epoch)
    writer.add_scalar("Accuracy/test", test_accuracies.mean(), epoch)
    
    return test_losses.mean(), test_accuracies.mean()
        
def is_best(best_acc, test_acc):
    return best_acc < test_acc
    

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    EPOCHS = config['training'][0]['epochs']
    patience = config['training'][0]['patience']
    
    model, optimizer, scheduler, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(config)

    model.to(device)

    #optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    val_dataloader = DataLoader(valid_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S").replace("-", "_").replace(":", "-")
    writer = SummaryWriter(log_dir=args.tensorboard_log_path + "/{}_{}_{}".format(model.name, train_dataset.name, formatted_time))
    
    best_acc = 0
    not_improved = 0
    for epoch in range(1,EPOCHS+1):
        train_losses, train_accuracies, val_losses, val_accuracies = train(epoch, train_dataloader, val_dataloader, writer)
        test_losses, test_accuracies = test(epoch, test_dataloader, writer)
        
        state_dict = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'Val Acc' : val_accuracies,
                'Test Acc' : test_accuracies
            }
        
        if(epoch % 5 == 0):
            print(f"Epoch [{epoch}/{EPOCHS}] : Valid Acc {val_accuracies:.4f}, Test Acc {test_accuracies:.4f}")
            torch.save(state_dict, args.model_saving_path + f"/latest_{model.name}.pth.tar")
        
        if(is_best(best_acc, test_accuracies)):
            best_acc = test_accuracies
            torch.save(state_dict, args.model_saving_path + "/model_best.pth.tar")
            not_improved = 0
        else:
            not_improved += 1
            
        if(not_improved > patience):
            print(f"Testing results not improving for {patience} Epochs, Stop training")
            break
    
    writer.close()