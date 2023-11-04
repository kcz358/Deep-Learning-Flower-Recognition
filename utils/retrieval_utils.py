
import faiss
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


def encode_database(dataset : Dataset, model : nn.Module, embedding_size : int, device : torch.device):
    """_summary_

    Args:
        dataset (Dataset): The Dataset that you want to encode into embeddings
        model (nn.Module): The embedding model that you use
        embedding_size (int): The embedding size of the vector
        device (torch.device): The device that you use

    Returns:
        The feature vector of the dataset, in shape (N, embedding_size)
    """    
    
    features = torch.zeros(len(dataset), embedding_size, dtype=torch.float32)
    dataloader = DataLoader(dataset, batch_size=32)
    model.eval()
    
    for iterations, (img, labels) in enumerate(dataloader, 1):
        img = img.to(device)
        output, embeddings = model(img)
        
        features[(iterations - 1) * 32 : iterations * 32, :] = embeddings.detach().cpu()
        
        if(iterations % 50 == 0):
            print(f"Finish Encoding [{iterations}/{len(dataloader)}]")
    
    return features.numpy()

def evaluate(db_features : np.array, q_features : np.array, db_labels : np.array, q_labels : np.array):
    print('====> Building faiss index')
    faiss_index = faiss.IndexFlatL2(db_features.shape[1])
    #faiss_index = faiss.IndexFlatIP(cfg.dsc_dim)
    faiss_index.add(db_features)

    print('====> Calculating recall @ N')
    n_values = [1,5,10,20]

    _, predictions = faiss_index.search(q_features, max(n_values))
    
    
    correct_at_n = np.zeros(len(n_values))
    #TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i,n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            db_label = db_labels[pred[:n]]
            if np.any(np.in1d(db_label, q_labels[qIx])):
                correct_at_n[i:] += 1
                break
    recall_at_n = correct_at_n / len(q_features)
    recalls = {} #make dict for output
    for i,n in enumerate(n_values):
        recalls[n] = recall_at_n[i]
        print("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))     
    return recalls