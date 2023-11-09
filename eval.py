import argparse
from glob import glob
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from utils import build_from_config, encode_database, evaluate

def parse_arguments():
    parser = argparse.ArgumentParser(description="A script that takes dataset path, model saving path, and tensorboard log path as arguments.")

    # Add arguments
    parser.add_argument('--checkpoint_path', '-p', required=True, help='Path to save the model')
    parser.add_argument("--config", "-c", required=True, type=str, help="Path to the config file.")

    return parser.parse_args() 

def acc_test(epoch, test_dataloader, writer=None):
    model.eval()
    test_losses = []
    test_accuracies = []
    predictions = []
    for iterations, (images, labels) in enumerate(test_dataloader, 1):
        images = images.to(device)
        labels = labels.to(device)
        
        output = model(images)
        #output[0] = F.softmax(output[0], dim = 1)
        loss = criterion(output, labels)
        
        test_losses.append(loss.item())
        
        pred = torch.argmax(output[0], dim=1)
        predictions += pred.tolist()
        acc = (pred == labels).sum() / len(labels)
        test_accuracies.append(acc)
        if iterations % 10 == 0:
            print(f"[{iterations}/{len(test_dataloader)}] Batch Acc : {acc}")
    
    test_losses = torch.tensor(test_losses, dtype=torch.float32)
    test_accuracies = torch.tensor(test_accuracies, dtype=torch.float32)
    if writer is not None:
        writer.add_scalar("Loss/test",test_losses.mean(), epoch)
        writer.add_scalar("Accuracy/test", test_accuracies.mean(), epoch)
    
    return test_losses.mean(), test_accuracies.mean(), predictions

def retrieval_test(epoch, test_dataset, train_dataset, writer=None):
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
    recalls, predictions = evaluate(db_features, q_features, db_labels, q_labels)
    
    if writer is not None:
        for i,n in recalls.items():
            #print("====> Recall@{}: {:.4f}".format(i, n))     
            writer.add_scalar('Val/Recall@' + str(i), n, epoch)
    return recalls, predictions, q_labels

if __name__ == "__main__":
    args = parse_arguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)  
        
    ckpt_files = glob(args.checkpoint_path + "/*")
    
    results = {}
    for files in ckpt_files:
        print(f"Evaluating {files}")
        ckpt = torch.load(files, map_location=device)
        try:
            model, _, _, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(ckpt['training_config'])
        except:
            print("Config not found in the checkpoint, use config given")
            model, _, _, criterion, train_dataset, valid_dataset, test_dataset = build_from_config(config)
        model.load_state_dict(ckpt['state_dict'])
        
        model.to(device)
        model.eval()
        
        test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        test_losses, test_accuracies, acc_predictions = acc_test(0, test_dataloader, None)
        test_recalls, predictions, q_labels = retrieval_test(0, test_dataset, train_dataset, None)
        print(f"Acc : {test_accuracies}")
        name = files.split('/')[-1]
        results[name] = {}
        results[name]['Test Acc'] = test_accuracies.item()
        results[name]['recalls'] = test_recalls
        results[name]['match_index'] = predictions.astype(int).tolist()
        results[name]['predictions'] = acc_predictions
        results[name]['gt'] = q_labels.astype(int).tolist()
    with open("./eval_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
        
        
        