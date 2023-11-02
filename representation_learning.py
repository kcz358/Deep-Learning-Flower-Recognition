import argparse
import datetime
import yaml

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from utils import build_from_config, encode_database, evaluate

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
    for iterations, (images, labels) in enumerate(train_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        #output[0] = F.softmax(output[0], dim = 1)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        
        if(iterations % 100 == 0):
            print(f"Batch [{iterations}/{len(train_dataloader)}] : Training Loss {sum(train_losses) / len(train_losses)}")
    
        
    train_losses = torch.tensor(train_losses, dtype=torch.float32)

    writer.add_scalar("Loss/train", train_losses.mean(), epoch)
    
    scheduler.step()
    return train_losses.mean()

def test(epoch, test_dataset, train_dataset, writer):
    db_features = encode_database(train_dataset, model, model.embedding_size, device)
    q_features = encode_database(test_dataset, model, model.embedding_size, device)
    
    db_labels = []
    for idx in range(len(train_dataset)):
        db_labels.append(train_dataset[idx][1])
    db_labels = np.array(db_labels, dtype=int)
    q_labels = []
    for idx in range(len(test_dataset)):
        q_labels.append(test_dataset[idx][1])
    q_labels = np.array(q_labels, dtype=int)
    
    # recalls a dict (k, recall@k)
    recalls = evaluate(db_features, q_features, db_labels, q_labels)
    
    for i,n in recalls.items():
        #print("====> Recall@{}: {:.4f}".format(i, n))     
        writer.add_scalar('Val/Recall@' + str(i), n, epoch)
    return recalls
        
def is_best_recall(recalls, best_score):
    if recalls[1] > best_score[1]:
        return True
    elif recalls[1] < best_score[1]:
        return False
    else:
        if recalls[5] > best_score[5]:
            return True
        elif recalls[5] < best_score[5]:
            return False
        else:
            return recalls[10] > best_score[10]
    

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
    
    best_acc = {
        1 : 0,
        5 : 0,
        10 : 0,
        20 : 0
    }
    not_improved = 0
    
    continue_ckpt = config['training'][0].get('continue_ckpt', None)
    if continue_ckpt is not None:
        ckpt = torch.load(continue_ckpt)
        model.load_state_dict(ckpt['state_dict'], strict=False)
        
    for epoch in range(1,EPOCHS+1):
        if(epoch == 1):
            print("Off the shelf model")
            test(0, test_dataset, train_dataset, writer)
            state_dict = {
                'epoch' : 0,
                'state_dict' : model.state_dict(),
                'Recalls' : test_recalls,
            }
            
            torch.save(state_dict, args.model_saving_path + f"/model_best.pth.tar")
            
        print(f"Epoch {epoch} ===>")
        train_losses = train(epoch, train_dataloader, val_dataloader, writer)
        test_recalls = test(epoch, test_dataset, train_dataset, writer)
        
        state_dict = {
                'epoch' : epoch,
                'state_dict' : model.state_dict(),
                'Recalls' : test_recalls,
            }
        
        if(epoch % 5 == 0):
            torch.save(state_dict, args.model_saving_path + f"/latest_{model.name}.pth.tar")
        
        if(is_best_recall(best_acc, test_recalls)):
            best_acc = test_recalls
            torch.save(state_dict, args.model_saving_path + "/model_best.pth.tar")
            not_improved = 0
        else:
            not_improved += 1
            
        if(not_improved > patience):
            print(f"Testing results not improving for {patience} Epochs, Stop training")
            break
    
    writer.close()