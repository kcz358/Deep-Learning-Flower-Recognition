import argparse
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets import flowers102
from models import resnet50

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that takes dataset path, model saving path, and tensorboard log path as arguments.")

    # Add arguments
    parser.add_argument('--dataset_path', required=True, help='Path to the dataset')
    parser.add_argument('--model_saving_path', required=True, help='Path to save the model')
    parser.add_argument('--tensorboard_log_path', required=True, help='Path to save TensorBoard logs')

    return parser.parse_args()   

def train(epoch, train_dataloader, val_dataloader, writer):
    model.train()
    train_losses = []
    train_accuracies = []
    for iterations, (images, labels) in enumerate(train_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        writer.add_scalar("Loss/train", epoch, loss.item())
        
        pred = torch.argmax(F.softmax(loss, dim = 0))
        acc = (pred == labels).sum()
        train_accuracies.append(acc)
        writer.add_scalar("Accuracy/train", epoch, acc)
        
        
    
    model.eval()
    val_losses = []
    val_accuracies = []
    for iterations, (images, labels) in enumerate(val_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        
        val_losses.append(loss.item())
        writer.add_scalar("Loss/val", epoch, loss.item())
        
        pred = torch.argmax(F.softmax(loss, dim = 0))
        acc = (pred == labels).sum()
        val_accuracies.append(acc)
        writer.add_scalar("Accuracy/val", epoch, acc)
    train_losses = torch.tensor(train_losses, dtype=torch.float32)
    train_accuracies = torch.tensor(train_accuracies, dtype=torch.float32)
    val_losses = torch.tensor(val_losses, dtype=torch.float32)
    val_accuracies = torch.tensor(val_accuracies, dtype=torch.float32)
    return train_losses.mean(), train_accuracies.mean(), val_losses.mean(), val_accuracies.mean()

def test(epoch, test_dataloader, writer):
    model.eval()
    test_losses = []
    test_accuracies = []
    for iterations, (images, labels) in enumerate(test_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        loss = criterion(output, labels)
        
        test_losses.append(loss.item())
        writer.add_scalar("Loss/test", epoch, loss.item())
        
        pred = torch.argmax(F.softmax(loss, dim = 0))
        acc = (pred == labels).sum()
        test_accuracies.append(acc)
        writer.add_scalar("Accuracy/test", epoch, acc)
    
    test_losses = torch.tensor(test_losses, dtype=torch.float32)
    test_accuracies = torch.tensor(test_accuracies, dtype=torch.float32)
    
    return test_losses.mean(), test_accuracies.mean()
        
        
    

if __name__ == '__main__':
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_dataset = flowers102.Flowers102(dataset_path=args.dataset_path, split='train')
    val_dataset = flowers102.Flowers102(dataset_path=args.dataset_path, split='valid')
    test_dataset = flowers102.Flowers102(dataset_path=args.dataset_path, split='test')
    
    model = resnet50.ResNet50(model_name="ResNet50", 
                              num_classes=train_dataset.num_classes, 
                              weights='DEFAULT')
    model.to(device)
    
    train_dataset.transform = model.transformation
    val_dataset.transform = model.transformation
    test_dataset.transform = model.transformation
    
    optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    train_dataloader = DataLoader(train_dataset, batch_size=64)
    val_dataloader = DataLoader(val_dataset, batch_size=64)
    test_dataloader = DataLoader(test_dataset, batch_size=64)
    
    criterion = nn.CrossEntropyLoss()
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d-%H:%M:%S").replace("-", "_").replace(":", "-")
    writer = SummaryWriter(log_dir=args.tensorboard_log_path + "/{}_{}_{}".format(model.name, train_dataset.name, formatted_time))
    
    for epoch in range(1,51):
        train_losses, train_accuracies, val_losses, val_accuracies = train(epoch, train_dataloader, val_dataloader, writer)
        test_losses, test_accuracies = test(epoch, test_dataloader, writer)
        
        if(epoch % 5 == 0):
            print(f"Epoch [{epoch}/{50}] : Valid Acc {val_accuracies}, Test Acc {test_accuracies}")
            state_dict = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'Val Acc' : val_accuracies,
                'Test Acc' : test_accuracies
            }
            torch.save(state_dict, args.model_saving_path)
    
    writer.close()