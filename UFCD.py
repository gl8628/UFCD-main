
import torch
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.cluster import KMeans
from multiprocessing import Pool
import os
import time
import leidenalg as la 
import igraph as ig
from collections import defaultdict, Counter, deque
import random
import multiprocessing as mp
from functools import partial
import shutil
import warnings
from sklearn.exceptions import ConvergenceWarning
import pickle
import scipy.sparse as sp
import math
from scipy.sparse.linalg import eigsh
from munkres import Munkres
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn import metrics
from scipy.linalg import orthogonal_procrustes
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
from pickle import UnpicklingError
import networkx.algorithms.community as nx_comm
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore", category=ConvergenceWarning)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.utils import to_undirected
from pathlib import Path

import os


from dhg.data import BlogCatalog


# ==================== Hyperparameters ====================
HIDDEN_DIM = 64 # Changed to 64 for better performance
NUM_PROCESSES = mp.cpu_count()
TEMP_DIR = "temp_embeddings"
RANDOM_SEED = 42
EPOCHS = 600
MESSAGE_PASSING_ITERATIONS = 500
cut_blocknum=5000
os.environ["PYTHONHASHSEED"] = str(RANDOM_SEED)
R=-0.05
# R=-0.10
D=6
OVERLAP_RATIO=0.1
MIN_BLOCK_SIZE=10

# Fix random seeds
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

import hashlib

def deterministic_hash(node):
    return int(hashlib.md5(str(node).encode()).hexdigest(), 16)


def init_worker(seed):
    import os, random, numpy as np, torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def cluster_acc(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_true = y_true - y_true.min()
    y_pred = y_pred - y_pred.min()
    
    D = max(y_true.max(), y_pred.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_true[i], y_pred[i]] += 1
    
    row_ind, col_ind = linear_sum_assignment(w.max() - w)
    mapping = {col: row for row, col in zip(row_ind, col_ind)}
    
    for j in range(D):
        if j not in mapping:
            used = set(mapping.values())
            for r in range(D):
                if r not in used:
                    mapping[j] = r
                    break
            else:
                mapping[j] = 0
    
    y_pred_mapped = np.array([mapping[y] for y in y_pred])
    acc = accuracy_score(y_true, y_pred_mapped)
    f1 = f1_score(y_true, y_pred_mapped, average='macro', zero_division=0)
    return acc, f1

from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score, f1_score

# ==================== GAT/HACD Models ====================



class GAT_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(GAT_Encoder, self).__init__()
        self.conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.conv2 = GATConv(hidden_dim*4, hidden_dim, heads=1, concat=False, dropout=dropout)
    
    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class GCN_Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(GCN_Encoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim*2)
        self.conv2 = GCNConv(hidden_dim*2, hidden_dim)
    
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    




class DualGatedEncoder(nn.Module):
    """
    双编码器软门控方案（方案A）
    - 包含完整的GCN和GAT两个独立编码器
    - gate_weight控制两个编码器输出的融合比例
    - gate_weight = 0 → 纯GCN输出
    - gate_weight = 1 → 纯GAT输出
    - 中间值 → 加权融合，平滑过渡
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.5):
        super(DualGatedEncoder, self).__init__()
        
        # GCN分支（完整的GCN编码器）
        self.gcn_conv1 = GCNConv(input_dim, hidden_dim * 2)
        self.gcn_conv2 = GCNConv(hidden_dim * 2, hidden_dim)
        
        # GAT分支（完整的GAT编码器，与原始GAT_Encoder结构一致）
        self.gat_conv1 = GATConv(input_dim, hidden_dim, heads=4, dropout=dropout)
        self.gat_conv2 = GATConv(hidden_dim * 4, hidden_dim, heads=1, concat=False, dropout=dropout)
        
        self.dropout = dropout
    
    def forward(self, x, edge_index, gate_weight=0.0):
        """
        Args:
            gate_weight: 门控权重 [0, 1]
                0 → 纯GCN
                1 → 纯GAT
                中间值 → GCN输出*(1-gate) + GAT输出*gate
        """
        # GCN分支前向传播
        x_gcn = F.relu(self.gcn_conv1(x, edge_index))
        x_gcn = F.dropout(x_gcn, p=self.dropout, training=self.training)
        x_gcn = self.gcn_conv2(x_gcn, edge_index)
        
        if gate_weight < 1e-8:
            return x_gcn  # 纯GCN，跳过GAT计算节省资源
        
        # GAT分支前向传播
        x_gat = F.elu(self.gat_conv1(x, edge_index))
        x_gat = F.dropout(x_gat, p=self.dropout, training=self.training)
        x_gat = self.gat_conv2(x_gat, edge_index)
        
        if gate_weight > 1 - 1e-8:
            return x_gat  # 纯GAT
        
        # 加权融合
        x_out = (1 - gate_weight) * x_gcn + gate_weight * x_gat
        return x_out


class SA_Attention(nn.Module):
    """Semantic Attention for meta-path based aggregation"""
    def __init__(self, in_size, hidden_size=128):
        super(SA_Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False),
        )

    def forward(self, z, h):
        w = self.project(z).mean(0)  # (M, 1)
        beta = torch.softmax(w, dim=0)  # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape)  # (N, M, 1)
        return (beta * z).sum(1)  # (N, D * K)
    


def compute_gate_weight_smooth(assortativity, avg_deg, R_low=-0.10, R_high=0.0, D_thresh=6):
    """
    计算平滑门控权重（方案A：基于图统计特征的固定门控）
    
    策略：
    - 当 avg_deg >= D_thresh 时：gate=0（高同配性/高度数，用GCN）
    - 当 assortativity <= R_low 且 avg_deg < D_thresh 时：gate=1（低同配性/低度数，用GAT）
    - 当 assortativity >= R_high 时：gate=0（高同配性，用GCN）
    - R_low < assortativity < R_high 时：线性过渡
    
    Args:
        assortativity: 图的同配性系数
        avg_deg: 平均度数
        R_low: 同配性下限阈值（低于此值完全用GAT）
        R_high: 同配性上限阈值（高于此值完全用GCN）
        D_thresh: 度数阈值（高于此值强制用GCN）
    
    Returns:
        gate_weight: [0, 1] 之间的门控权重
    """
    # 度数门控：平均度 >= D_thresh 时强制用 GCN
    if avg_deg >= D_thresh:
        return 0.0
    
    # 同配性门控
    if assortativity <= R_low:
        return 1.0  # 低同配性，完全用GAT
    elif assortativity >= R_high:
        return 0.0  # 高同配性，完全用GCN
    else:
        # 线性过渡：从 R_low 处的 1.0 线性降到 R_high 处的 0.0
        ratio = (assortativity - R_low) / (R_high - R_low)
        return 1.0 - ratio


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, BatchNorm
from sklearn.preprocessing import normalize
import numpy as np



def negative_sampling(edge_index, num_nodes, num_neg_samples, device):
    """
    高效负采样：生成与现有边不重复的随机边。
    """
    # 将现有边转换为集合（加速判断）
    existing_edges = set((edge_index[0,i].item(), edge_index[1,i].item()) for i in range(edge_index.size(1)))
    neg_edges = []
    while len(neg_edges) < num_neg_samples:
        src = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
        dst = torch.randint(0, num_nodes, (num_neg_samples,), device=device)
        # 过滤掉自环和已有边
        mask = (src != dst)
        for i in range(num_neg_samples):
            if mask[i]:
                s, d = src[i].item(), dst[i].item()
                if (s, d) not in existing_edges and (d, s) not in existing_edges:
                    neg_edges.append((s, d))
                    if len(neg_edges) == num_neg_samples:
                        break
    neg_src = torch.tensor([e[0] for e in neg_edges], device=device)
    neg_dst = torch.tensor([e[1] for e in neg_edges], device=device)
    return torch.stack([neg_src, neg_dst])



#4.28
def get_hacd_embeddings(edge_index, x, num_epochs=200, lr=0.01, device='cpu',  verbose=True,encoder_type="gcn", hidden_dim=None):
    """Get HACD embeddings using link prediction loss"""
    if hidden_dim is None:
        hidden_dim = HIDDEN_DIM
    input_dim = x.shape[1]
    n_nodes = x.shape[0]
    

    if encoder_type == 'gat':
        model = GAT_Encoder(input_dim, hidden_dim).to(device)
    elif encoder_type == 'gcn':
        model = GCN_Encoder(input_dim, hidden_dim).to(device)
  
 
   

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    edge_index = edge_index.to(device)
    x = x.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        z = model(x, edge_index)
        
        # Link prediction loss: positive + negative samples
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        # Positive loss (existing edges)
        pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
        
        # Negative loss (random non-edges)
        neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1)), device=device)
        neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}")
    
    if verbose:
        print(f"Final loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        emb = model(x, edge_index)
        return emb.cpu().numpy()



def get_hacd_embeddings_soft_gate(edge_index, x, num_epochs=200, lr=0.01, device='cpu',
                                  verbose=True, hidden_dim=None, gate_weight=0.0):
    """
    软门控版本的HACD嵌入学习（方案A）
    
    改进：当 gate_weight=0 或 1 时，直接使用独立的 GCN/GAT 编码器，
    避免 DualGatedEncoder 中未使用分支的参数初始化差异影响训练。
    仅在 0 < gate_weight < 1 时使用 DualGatedEncoder 进行融合。
    """
    if hidden_dim is None:
        hidden_dim = HIDDEN_DIM
    input_dim = x.shape[1]
    n_nodes = x.shape[0]
    
    # 判断是否为纯编码器模式（避免未使用参数的初始化差异）
    use_pure_encoder = (gate_weight < 1e-8) or (gate_weight > 1 - 1e-8)
    
    if use_pure_encoder:
        # 纯模式：直接使用独立编码器，参数初始化与 Embed-64 完全一致
        if gate_weight < 1e-8:
            model = GCN_Encoder(input_dim, hidden_dim).to(device)
            encoder_label = "pure-GCN"
        else:
            model = GAT_Encoder(input_dim, hidden_dim).to(device)
            encoder_label = "pure-GAT"
        if verbose:
            print(f"  [SoftGate] Using {encoder_label} (gate={gate_weight:.3f})")
    else:
        # 融合模式：使用双编码器
        model = DualGatedEncoder(input_dim, hidden_dim).to(device)
        encoder_label = f"dual-gate({gate_weight:.3f})"
        if verbose:
            print(f"  [SoftGate] Using DualGatedEncoder (gate={gate_weight:.3f})")
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    edge_index = edge_index.to(device)
    x = x.to(device)
    
    model.train()
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        if use_pure_encoder:
            z = model(x, edge_index)
        else:
            z = model(x, edge_index, gate_weight=gate_weight)
        
        # Link prediction loss: positive + negative samples
        adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
        # Positive loss (existing edges)
        pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
        
        # Negative loss (random non-edges)
        neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1)), device=device)
        neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
        
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if verbose and epoch % 50 == 0:
            print(f"Epoch {epoch}, loss: {loss.item():.4f}, gate: {gate_weight:.3f}")
    
    if verbose:
        print(f"Final loss: {loss.item():.4f}")
    
    model.eval()
    with torch.no_grad():
        if use_pure_encoder:
            emb = model(x, edge_index)
        else:
            emb = model(x, edge_index, gate_weight=gate_weight)
        return emb.cpu().numpy()



# def get_hacd_embeddings(edge_index, x, num_epochs=200, lr=0.01, device='cpu', verbose=True, encoder_type="gcn"):
#     """Get HACD embeddings using link prediction loss"""
#     input_dim = x.shape[1]
#     n_nodes = x.shape[0]
    
#     if encoder_type == 'gat':
#         model = GAT_Encoder(input_dim, HIDDEN_DIM).to(device)
#     elif encoder_type == 'gcn':
#         model = GCN_Encoder(input_dim, HIDDEN_DIM).to(device)
#     else:
#         model = HACD_Embedder(input_dim, HIDDEN_DIM, HIDDEN_DIM).to(device)
    
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
#     edge_index = edge_index.to(device)
#     x = x.to(device)
    
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         z = model(x, edge_index)
        
#         # Link prediction loss
#         adj_pred = torch.sigmoid(torch.mm(z, z.t()))
        
#         # Positive loss
#         pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
        
#         # Negative loss - 增加负采样数量（2倍）
#         neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1) * 2), device=device)
#         neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
        
#         loss = pos_loss + neg_loss
#         loss.backward()
#         optimizer.step()
        
#         if verbose and epoch % 50 == 0:
#             print(f"Epoch {epoch}, loss: {loss.item():.4f}")
    
#     if verbose:
#         print(f"Final loss: {loss.item():.4f}")
    
#     model.eval()
#     with torch.no_grad():
#         emb = model(x, edge_index)
#         return emb.cpu().numpy()


# #5.17
# def get_hacd_embeddings(edge_index, x, num_epochs=200, lr=0.01, device='cpu', 
#                         use_meta_paths=False, verbose=True, encoder_type="gcn"):
#     """Get HACD embeddings using link prediction loss with early stopping and LR scheduling"""
#     input_dim = x.shape[1]
#     n_nodes = x.shape[0]
    
#     # 根据 encoder_type 选择模型
#     if encoder_type == 'gat':
#         model = GAT_Encoder(input_dim, HIDDEN_DIM).to(device)
#     elif encoder_type == 'gcn':
#         model = GCN_Encoder(input_dim, HIDDEN_DIM).to(device)
#     else:
#         model = HACD_Embedder(input_dim, HIDDEN_DIM, HIDDEN_DIM).to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    
#     # 学习率调度器：当 loss 停滞时降低学习率
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
#     )
    
#     edge_index = edge_index.to(device)
#     x = x.to(device)
    
#     model.train()
#     best_loss = float('inf')
#     patience_counter = 0
#     patience = 15          # 连续 patience 个 epoch 无改善则停止
#     min_epochs = 50        # 至少训练 min_epochs 轮，避免过早停止
    
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         z = model(x, edge_index)
        
#         # Link prediction loss
#         adj_pred = torch.sigmoid(torch.mm(z, z.t()))
#         pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
#         neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1)), device=device)
#         neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
#         loss = pos_loss + neg_loss
        
#         loss.backward()
#         optimizer.step()
#         scheduler.step(loss)   # 根据当前 loss 调整学习率
        
#         if verbose and epoch % 50 == 0:
#             print(f"Epoch {epoch}, loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.6f}")
        
#         # 早停逻辑（仅当 epoch >= min_epochs）
#         if epoch >= min_epochs:
#             if loss.item() < best_loss - 1e-4:
#                 best_loss = loss.item()
#                 patience_counter = 0
#             else:
#                 patience_counter += 1
#                 if patience_counter >= patience:
#                     if verbose:
#                         print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.4f}")
#                     break
#     else:
#         # 正常完成所有 epoch
#         if verbose:
#             print(f"Completed all {num_epochs} epochs, final loss: {loss.item():.4f}")
    
#     if verbose:
#         print(f"Final loss: {loss.item():.4f}")
    
#     model.eval()
#     with torch.no_grad():
#         emb = model(x, edge_index)
#         return emb.cpu().numpy()


# def get_hacd_embeddings(edge_index, x, num_epochs=200, lr=0.01, device='cpu',
#                         encoder_type='gcn', verbose=True):
#     input_dim = x.shape[1]
#     n_nodes = x.shape[0]
    
#     # 选择模型
#     if encoder_type == 'gat':
#         model = GAT_Encoder(input_dim, HIDDEN_DIM).to(device)
#     else:
#         model = GCN_Encoder(input_dim, HIDDEN_DIM).to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
#     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
#     edge_index = edge_index.to(device)
#     x = x.to(device)
    
#     # 划分验证集（固定随机种子）
#     val_size = max(100, int(n_nodes * 0.2))
#     val_idx = torch.randperm(n_nodes, device=device)[:val_size]
    
#     best_sil = -1
#     best_state = None
#     patience = 15
#     wait = 0
#     min_epochs = 50
    
#     model.train()
#     for epoch in range(num_epochs):
#         optimizer.zero_grad()
#         z = model(x, edge_index)
#         adj_pred = torch.sigmoid(torch.mm(z, z.t()))
#         pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
#         neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1)), device=device)
#         neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
#         loss = pos_loss + neg_loss
#         loss.backward()
#         optimizer.step()
#         scheduler.step(loss)
        
#         if verbose and epoch % 50 == 0:
#             print(f"Epoch {epoch}, loss={loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}")
        
#         # 每隔10个epoch评估验证集轮廓系数
#         if epoch >= min_epochs and epoch % 10 == 0:
#             model.eval()
#             with torch.no_grad():
#                 z_full = model(x, edge_index)
#                 z_val = z_full[val_idx].cpu().numpy()
#             from sklearn.cluster import KMeans
#             from sklearn.metrics import silhouette_score
#             k = min(15, val_size-1)  # 固定k，也可用真实类别数但这里无监督
#             kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
#             labels = kmeans.fit_predict(z_val)
#             sil = silhouette_score(z_val, labels) if len(set(labels)) > 1 else -1
#             model.train()
            
#             if sil > best_sil:
#                 best_sil = sil
#                 best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
#                 wait = 0
#             else:
#                 wait += 1
#                 if wait >= patience:
#                     if verbose:
#                         print(f"Early stopping at epoch {epoch}, best silhouette={best_sil:.4f}")
#                     model.load_state_dict(best_state)
#                     break
#     else:
#         if verbose:
#             print(f"Completed {num_epochs} epochs, final silhouette={best_sil:.4f}")
    
#     model.eval()
#     with torch.no_grad():
#         emb = model(x, edge_index).cpu().numpy()
#     return emb

def auto_select_encoder(edge_index, x, num_epochs=50, device='cpu', verbose=True):
    n_nodes = x.shape[0]
    val_size = max(100, int(n_nodes * 0.2))
    val_idx = torch.randperm(n_nodes, device=device)[:val_size]
    scores = {}
    for enc in ['gcn', 'gat']:
        if enc == 'gcn':
            model = GCN_Encoder(x.shape[1], HIDDEN_DIM).to(device)
        else:
            model = GAT_Encoder(x.shape[1], HIDDEN_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        edge_index = edge_index.to(device)
        x_dev = x.to(device)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            z = model(x_dev, edge_index)
            adj_pred = torch.sigmoid(torch.mm(z, z.t()))
            pos_loss = -torch.log(adj_pred[edge_index[0], edge_index[1]] + 1e-8).mean()
            neg_indices = torch.randint(0, n_nodes, (2, edge_index.size(1)), device=device)
            neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            z_full = model(x_dev, edge_index)
            z_val = z_full[val_idx].cpu().numpy()
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        k = min(15, val_size-1)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(z_val)
        sil = silhouette_score(z_val, labels) if len(set(labels)) > 1 else -1
        scores[enc] = sil
        if verbose:
            print(f"Quick eval: {enc.upper()} silhouette={sil:.4f}")
    best = max(scores, key=scores.get)
    if verbose:
        print(f"Auto-selected encoder: {best.upper()}")
    return best

def auto_choose_clustering_method(embed_matrix, k):
    from scipy.sparse.linalg import svds
    try:
        U, S, Vt = svds(embed_matrix, k=k)
        ratio = np.sum(S) / np.linalg.norm(embed_matrix, ord='nuc')
        return 'gennci' if ratio > 0.7 else 'kmeans'
    except:
        return 'kmeans'

# ==================== Structural Community Detection (for non-attribute graphs) ====================

def generate_block_community_structural(args):
    """Pure structural Leiden community detection - no embeddings"""
    try:
        weighted_edge_block, block_nodes, block_id = args
        
        block_G = nx.Graph()
        if not weighted_edge_block.empty:
            valid_edges = weighted_edge_block[['u', 'v', 'weight']].values
            if len(valid_edges) > 0:
                block_G.add_weighted_edges_from(valid_edges)
        for node in block_nodes:
            if node not in block_G:
                block_G.add_node(node)
        
        if not block_G.nodes():
            return (block_id, {})
        
        num_nodes = len(block_G.nodes())
        num_edges = len(block_G.edges())
        
        def run_leiden():
            node_list = sorted(block_G.nodes())
            node_to_idx = {node: idx for idx, node in enumerate(node_list)}
            block_ig = ig.Graph(directed=False)
            block_ig.add_vertices(len(node_list))
            
            edges, edge_weights = [], []
            for u, v, data in block_G.edges(data=True):
                edges.append((node_to_idx[u], node_to_idx[v]))
                edge_weights.append(float(data.get('weight', 1.0)))
            if edges:
                block_ig.add_edges(edges)
                block_ig.es['weight'] = edge_weights
            
            if num_nodes > 500 and num_edges > 0:
                partition = la.find_partition(
                    block_ig, la.ModularityVertexPartition,
                    weights='weight', n_iterations=20, seed=42
                )
            else:
                partition = la.find_partition(
                    block_ig, la.ModularityVertexPartition,
                    weights='weight', n_iterations=10, seed=42
                )
            
            leiden_comm = np.array(partition.membership)
            return [leiden_comm[node_to_idx[node]] if node in node_to_idx else 0 
                    for node in block_nodes]
        
        block_fine_comm = run_leiden()
        global_comm_prefix = block_id * 1000000
        block_comm_dict = {node: global_comm_prefix + comm 
                          for node, comm in zip(block_nodes, block_fine_comm)}
        
        return (block_id, block_comm_dict)
    
    except Exception as e:
        print(f"Block {block_id} structural community detection failed: {str(e)}")
        block_comm_dict = {node: block_id * 1000000 + i for i, node in enumerate(block_nodes)}
        return (block_id, block_comm_dict)


def build_lightweight_graph(edge_df, nodes):
    """Build lightweight graph using sparse matrix"""
    all_nodes = list(nodes)
    node_to_idx = {node: i for i, node in enumerate(all_nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}
    n_nodes = len(all_nodes)
    
    rows, cols, data = [], [], []
    for u, v in edge_df[['u', 'v']].values:
        if u in node_to_idx and v in node_to_idx:
            i, j = node_to_idx[u], node_to_idx[v]
            rows.append(i); cols.append(j); data.append(1.0)
            rows.append(j); cols.append(i); data.append(1.0)
    
    adj_matrix = sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
    
    class LightweightGraph:
        def __init__(self, adj, idx_to_node, node_to_idx):
            self.adj = adj
            self.idx_to_node = idx_to_node
            self.node_to_idx = node_to_idx
            self._nodes = set(idx_to_node.values())
        
        def __contains__(self, node):
            return node in self.node_to_idx
        
        def neighbors(self, node):
            idx = self.node_to_idx[node]
            col_indices = self.adj[idx].nonzero()[1]
            return [self.idx_to_node[j] for j in col_indices]
        
        def has_node(self, node):
            return node in self.node_to_idx
        
        def nodes(self):
            return self._nodes
        
        def number_of_nodes(self):
            return len(self._nodes)
        
        def degree(self, node):
            idx = self.node_to_idx[node]
            return self.adj[idx].nnz
        
        def edges(self, data=False):
            coo = self.adj.tocoo()
            if data:
                for r, c, v in zip(coo.row, coo.col, coo.data):
                    if r <= c:
                        yield (self.idx_to_node[r], self.idx_to_node[c], {'weight': v})
            else:
                for r, c in zip(coo.row, coo.col):
                    if r <= c:
                        yield (self.idx_to_node[r], self.idx_to_node[c])
    
    return LightweightGraph(adj_matrix, idx_to_node, node_to_idx)


def global_optimization_with_overlap(G, comm_dict, all_nodes):
    """Global optimization considering overlapping nodes"""
    improved_comm_dict = comm_dict.copy()
    comm_sizes = defaultdict(int)
    for node, comm_id in improved_comm_dict.items():
        comm_sizes[comm_id] += 1
    
    for node in all_nodes:
        if node not in G:
            continue
            
        current_comm = improved_comm_dict.get(node, -1)
        if current_comm == -1:
            continue
            
        neighbors = list(G.neighbors(node))
        if not neighbors:
            continue
            
        neighbor_comms = defaultdict(int)
        for neighbor in neighbors:
            neighbor_comm = improved_comm_dict.get(neighbor, -1)
            if neighbor_comm != -1:
                neighbor_comms[neighbor_comm] += 1
        
        if neighbor_comms:
            best_comm = max(
                neighbor_comms.items(),
                key=lambda x: (x[1], comm_sizes.get(x[0], 0))
            )[0]
            
            current_conn = neighbor_comms.get(current_comm, 0)
            best_conn = neighbor_comms[best_comm]
            
            if best_comm != current_comm and best_conn > current_conn:
                comm_sizes[current_comm] = max(0, comm_sizes.get(current_comm, 0) - 1)
                comm_sizes[best_comm] = comm_sizes.get(best_comm, 0) + 1
                improved_comm_dict[node] = best_comm
    
    return improved_comm_dict


def merge_small_communities_fast(partition_dict, edge_df, min_size=3):
    """Fast merge small communities"""
    communities = partition_dict.copy()
    
    small_comms = {}
    large_comms = {}
    node_to_comm = {}
    
    for cid, nodes in communities.items():
        node_list = list(nodes)
        if len(node_list) < min_size:
            small_comms[cid] = node_list
        else:
            large_comms[cid] = node_list
        for node in node_list:
            node_to_comm[node] = cid
    
    if not small_comms:
        return communities
    
    comm_connections = defaultdict(lambda: defaultdict(int))
    for u, v in edge_df.values:
        comm_u = node_to_comm.get(u, -1)
        comm_v = node_to_comm.get(v, -1)
        
        if comm_u != -1 and comm_v != -1 and comm_u != comm_v:
            comm_connections[comm_u][comm_v] += 1
            comm_connections[comm_v][comm_u] += 1
    
    merged_result = {cid: set(nodes) for cid, nodes in large_comms.items()}
    comm_size = {cid: len(nodes) for cid, nodes in merged_result.items()}
    
    for small_cid, small_nodes in small_comms.items():
        connections = comm_connections.get(small_cid, {})
        
        candidate_large = {}
        for neighbor_comm, weight in connections.items():
            if neighbor_comm in merged_result:
                candidate_large[neighbor_comm] = weight
        
        if candidate_large:
            best_comm = max(
                candidate_large.items(),
                key=lambda x: (x[1], comm_size.get(x[0], 0))
            )[0]
        else:
            if comm_size:
                best_comm = min(comm_size.items(), key=lambda x: x[1])[0]
            else:
                best_comm = max(merged_result.keys(), default=-1) + 1
                merged_result[best_comm] = set()
                comm_size[best_comm] = 0
        
        if best_comm not in merged_result:
            merged_result[best_comm] = set()
            comm_size[best_comm] = 0
        
        merged_result[best_comm].update(small_nodes)
        comm_size[best_comm] += len(small_nodes)
    
    final_result = {}
    for new_id, (_, nodes) in enumerate(merged_result.items()):
        if nodes:
            final_result[new_id] = list(nodes)
    
    return final_result


def optimize_community_structure(node_to_community_dict, edge_df=None, min_size=3):
    """Optimize community structure"""
    community_to_nodes = defaultdict(list)
    for node, comm_id in node_to_community_dict.items():
        community_to_nodes[comm_id].append(node)
    
    merged_community_to_nodes = merge_small_communities_fast(community_to_nodes, edge_df, min_size)
    
    final_node_to_community = {}
    for comm_id, nodes in merged_community_to_nodes.items():
        for node in nodes:
            final_node_to_community[node] = comm_id
    
    return final_node_to_community


# ==================== Optimized Training Function ====================

# ========== 1. 随机游走采样（带重启）==========
def random_walk_sampling(G, target_nodes, restart_prob=0.15, max_steps=1000, seed=42):
    """
    使用带重启的随机游走从图中采样节点，保持局部社区结构。
    
    Parameters
    ----------
    G : nx.Graph
        原始图（节点为原始整数ID）。
    target_nodes : int
        期望采样的节点数量。
    restart_prob : float, default=0.15
        随机游走中重启到起始节点的概率。
    max_steps : int, default=1000
        每个游走的最大步数。
    seed : int, default=42
        随机种子。

    Returns
    -------
    sampled_nodes : list
        采样得到的节点列表（原始ID），长度 <= target_nodes。
    """
    random.seed(seed)
    np.random.seed(seed)
    
    if G.number_of_nodes() <= target_nodes:
        return list(G.nodes())
    
    nodes = list(G.nodes())
    degrees = np.array([G.degree(n) for n in nodes])
    probs = degrees / degrees.sum()
    start_node = np.random.choice(nodes, p=probs)
    
    sampled = set()
    current = start_node
    sampled.add(current)
    
    while len(sampled) < target_nodes:
        if random.random() < restart_prob:
            # 从已采样节点中随机选一个作为新起点
            current = random.choice(list(sampled))
        else:
            neighbors = list(G.neighbors(current))
            if not neighbors:
                current = random.choice(list(sampled))
                continue
            current = random.choice(neighbors)
        
        sampled.add(current)
        if len(sampled) >= target_nodes:
            break
        if len(sampled) > target_nodes * 1.5:
            break
    
    if len(sampled) < target_nodes:
        remaining = [n for n in G.nodes() if n not in sampled]
        needed = target_nodes - len(sampled)
        if remaining:
            sampled.update(random.sample(remaining, min(needed, len(remaining))))
    
    return list(sampled)

# ========== 2. 在采样子图上训练 HACD 嵌入（小图模式）==========
def train_hacd_on_subgraph(sub_G, node_features, epochs=100, hidden_dim=64, device='cpu', verbose=False):
    """
    在给定的子图（NetworkX图）上训练 HACD 编码器，返回节点嵌入。
    
    Parameters
    ----------
    sub_G : nx.Graph
        已重新编号为连续整数 0..n-1 的子图（建议使用 induced_subgraph 函数处理）。
    node_features : np.ndarray
        子图节点的原始特征矩阵，形状 (n, in_dim)，行顺序与 sub_G 节点编号一致。
    epochs : int
        训练轮数。
    hidden_dim : int
        最终嵌入维度。
    device : str
        'cpu' 或 'cuda'。
    verbose : bool
        是否打印训练过程。

    Returns
    -------
    emb : np.ndarray
        子图节点嵌入，形状 (n, hidden_dim)。
    """
    
    # 为了简单，直接复用已有的 get_hacd_embeddings 函数，但需要构造 edge_index 和 x_tensor
    n = sub_G.number_of_nodes()
    edges = list(sub_G.edges())
    if len(edges) == 0:
        return np.random.randn(n, hidden_dim).astype(np.float32)
    
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_index = to_undirected(edge_index)  # 转换为双向边
    x_tensor = torch.tensor(node_features, dtype=torch.float32)
    
    # 使用默认的 GCN 编码器（快速比较，可根据数据集特性选择，但为公平起见固定为 GCN）
    # 注意：原 get_hacd_embeddings 内部会重新实例化模型，我们直接调用
    emb = get_hacd_embeddings(edge_index, x_tensor, num_epochs=epochs, lr=0.01,
                              device=device, verbose=verbose, encoder_type='gcn',
                              hidden_dim=hidden_dim)
    return emb

# ========== 3. 构建诱导子图并重新编号 ==========
def induced_subgraph(G, nodes):
    """
    提取诱导子图，并将节点重新编号为 0..len(nodes)-1。
    
    Parameters
    ----------
    G : nx.Graph
        原始图。
    nodes : list
        原始节点 ID 列表。

    Returns
    -------
    sub_G : nx.Graph
        重新编号后的子图。
    mapping : dict
        原始节点 -> 新编号的映射。
    inv_mapping : dict
        新编号 -> 原始节点的映射。
    """
    sub_G = G.subgraph(nodes).copy()
    mapping = {old: new for new, old in enumerate(sub_G.nodes())}
    inv_mapping = {new: old for old, new in mapping.items()}
    sub_G = nx.relabel_nodes(sub_G, mapping, copy=True)
    return sub_G, mapping, inv_mapping

def auto_choose_feature_strategy(G, feat_matrix, num_classes=None, sample_ratio=0.1,
                                 max_sample_nodes=2000, fast=True, seed=42):
    """
    自动选择特征拼接策略：'feat', 'emb', 'both'
    
    参数
    ----
    G : nx.Graph
        原始图（节点连续编号 0..N-1）
    feat_matrix : np.ndarray (N, dim)
        原始特征矩阵（已 PCA 降维或 SRS 特征）
    num_classes : int, optional
        真实类别数，用于 KMeans 的 k
    sample_ratio : float
        采样节点比例（相对总节点数）
    max_sample_nodes : int
        最大采样节点数
    fast : bool
        若为 True，只比较 'feat' 和 'both'；否则比较三种
    seed : int
        随机种子
    
    返回
    ----
    strategy : str
        'feat', 'emb', 或 'both'
    scores : dict
        各策略在采样子图上的轮廓系数
    """
    random.seed(seed)
    np.random.seed(seed)
    
    N = G.number_of_nodes()
    S = min(int(N * sample_ratio), max_sample_nodes)
    if S < 50:
        # 图太小，直接使用 'feat' 作为默认（或运行全图比较）
        return 'feat', {}
    
    # 1. 采样节点（带重启的随机游走）
    sampled_nodes = random_walk_sampling(G, S, restart_prob=0.15, seed=seed)
    sub_G, mapping, inv_mapping = induced_subgraph(G, sampled_nodes)
    n_sub = sub_G.number_of_nodes()
    
    # 提取子图特征
    new_to_original = {new: old for old, new in mapping.items()}
    feat_sub = np.array([feat_matrix[new_to_original[i]] for i in range(n_sub)], dtype=np.float32)
    
    # 计算子图结构特征：度、聚类系数、PageRank
    deg = np.log1p([sub_G.degree(i) for i in range(n_sub)]).reshape(-1,1)
    clust = np.array([nx.clustering(sub_G, i) for i in range(n_sub)]).reshape(-1,1)
    pr = np.array(list(nx.pagerank(sub_G).values())).reshape(-1,1)
    
    # 2. 对于需要 HACD 嵌入的策略，先训练子图嵌入（100 轮）
    # 注意：即使只评估 'both'，也需要 hacd_emb，所以统一计算一次
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hacd_emb = train_hacd_on_subgraph(sub_G, feat_sub, epochs=100, hidden_dim=HIDDEN_DIM, device=device, verbose=False)
    
    # 3. 构建三种组合
    combined_feat = np.hstack([feat_sub, deg, clust, pr])
    combined_emb = np.hstack([hacd_emb, deg, clust, pr])
    combined_both = np.hstack([hacd_emb, feat_sub, deg, clust, pr])
    
    # 归一化（L2）
    combined_feat = normalize(combined_feat, norm='l2')
    combined_emb = normalize(combined_emb, norm='l2')
    combined_both = normalize(combined_both, norm='l2')
    
    # 4. 聚类并计算轮廓系数
    k = min(num_classes, n_sub-1) if num_classes and num_classes > 1 else min(15, n_sub-1)
    if k < 2:
        return 'feat', {}
    
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    def get_silhouette(combined):
        kmeans = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(combined)
        return silhouette_score(combined, kmeans.labels_)
    
    scores = {}
    scores['feat'] = get_silhouette(combined_feat)
    scores['both'] = get_silhouette(combined_both)
    if not fast:
        scores['emb'] = get_silhouette(combined_emb)
    
    # 5. 决策：选择轮廓系数最高的策略
    best = max(scores, key=scores.get)
    return best, scores

def train_single_block(args):
    """Optimized single block training - vectorized edge filtering"""
    block_id, block_nodes, weighted_edge_df, node_feat_dict, node_degree_dict, clustering, pagerank, HIDDEN_DIM, RANDOM_SEED,strategy,need_nom,use_cc = args
    
    block_node_set = set(block_nodes)
    block_edge_mask = np.array([u in block_node_set and v in block_node_set 
                                 for u, v in weighted_edge_df[['u', 'v']].values])
    block_edges = weighted_edge_df.iloc[np.where(block_edge_mask)[0]]
    
    if block_edges.empty:
        emb_dict = {node: np.random.normal(0, 0.1, HIDDEN_DIM).astype(np.float32) for node in block_nodes}
        return block_id, emb_dict
    
    block_node_list = list(block_nodes)
    node_to_idx = {node: i for i, node in enumerate(block_node_list)}
    feat_matrix = np.array([node_feat_dict[node] for node in block_node_list], dtype=np.float32)
    
    
    
    deg = np.log1p([node_degree_dict[node] for node in block_node_list]).reshape(-1,1)
    clust = np.array([clustering.get(node, 0) for node in block_node_list]).reshape(-1,1)
    pr_vals = np.array([pagerank.get(node, 0) for node in block_node_list]).reshape(-1,1)


    if not block_edges.empty:
            # 聚合 u 和 v 方向（假设边是单向存储，无向图每条边仅出现一次）
            wdeg_u = block_edges.groupby('u')['weight'].sum().to_dict()
            wdeg_v = block_edges.groupby('v')['weight'].sum().to_dict()
            wdeg_dict = {}
            for k, v in wdeg_u.items():
                wdeg_dict[k] = wdeg_dict.get(k, 0) + v
            for k, v in wdeg_v.items():
                wdeg_dict[k] = wdeg_dict.get(k, 0) + v
    else:
            wdeg_dict = {}
        
    weighted_deg = np.log1p([wdeg_dict.get(node, 0) for node in block_node_list]).reshape(-1, 1)

    if strategy=="emb" :

        edge_array = block_edges[['u', 'v']].values
        u_idx = np.array([node_to_idx.get(u, -1) for u in edge_array[:, 0]])
        v_idx = np.array([node_to_idx.get(v, -1) for v in edge_array[:, 1]])
        valid_mask = (u_idx >= 0) & (v_idx >= 0)
        valid_edges_u = u_idx[valid_mask]
        valid_edges_v = v_idx[valid_mask]
        
        if len(valid_edges_u) == 0:
            emb_dict = {node: np.random.normal(0, 0.1, HIDDEN_DIM).astype(np.float32) for node in block_nodes}
            return block_id, emb_dict
        
        edge_index_tensor = torch.tensor([valid_edges_u, valid_edges_v], dtype=torch.long).contiguous()
        edge_index_tensor = to_undirected(edge_index_tensor)
        
        x_tensor = torch.tensor(feat_matrix, dtype=torch.float32)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hacd_emb = get_hacd_embeddings(edge_index_tensor, x_tensor, num_epochs=EPOCHS, lr=0.01, device=device,  verbose=False, hidden_dim=HIDDEN_DIM)

        if use_cc:
            if strategy=="emb":
                combined = np.hstack([hacd_emb, weighted_deg, clust])
        else:
            if strategy=="emb":
                combined = np.hstack([hacd_emb])

        # else:
        #     combined = np.hstack([hacd_emb,feat_matrix, deg, clust])
        
    elif strategy=="feat":
        # combined = np.hstack([feat_matrixc])
        
        if use_cc:
            combined = np.hstack([feat_matrix,weighted_deg,clust])
        else:
            combined = np.hstack([feat_matrix])
        # combined = np.hstack([feat_matrix])

    
    
    # combined = np.hstack([hacd_emb, deg, clust, pr_vals])
    if need_nom:
        combined = normalize(combined, norm='l2')
    
    n_samples, n_features = combined.shape
    if n_features > HIDDEN_DIM:
        if n_samples > HIDDEN_DIM:
            pca_local = PCA(n_components=HIDDEN_DIM, random_state=RANDOM_SEED)
            combined = pca_local.fit_transform(combined)
        else:
            max_comp = max(1, n_samples - 1)
            if max_comp < HIDDEN_DIM:
                pca_local = PCA(n_components=max_comp, random_state=RANDOM_SEED)
                reduced = pca_local.fit_transform(combined)
                pad = np.zeros((reduced.shape[0], HIDDEN_DIM - max_comp), dtype=reduced.dtype)
                combined = np.hstack([reduced, pad])
            else:
                pca_local = PCA(n_components=HIDDEN_DIM, random_state=RANDOM_SEED)
                combined = pca_local.fit_transform(combined)
    elif n_features < HIDDEN_DIM:
        combined = np.pad(combined, ((0,0), (0, HIDDEN_DIM - n_features)), mode='constant')
    
    emb_dict = {node: combined[i] for i, node in enumerate(block_node_list)}
    return block_id, emb_dict


# ==================== Data Processing Functions ====================

def split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size):
    """BFS connectivity-based partitioning"""
    visited = set()
    adjacency = defaultdict(list)
    for u, v in edge_df[['u', 'v']].values:
        adjacency[u].append(v)
        adjacency[v].append(u)
    
    sorted_nodes = sorted(all_nodes, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
    blocks = []
    current_block = []
    
    for node in sorted_nodes:
        if node in visited:
            continue
        
        queue = deque([node])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            
            visited.add(current)
            current_block.append(current)
            if len(current_block) >= block_size:
                blocks.append(current_block)
                current_block = []
            
            neighbors = adjacency.get(current, [])
            neighbors_sorted = sorted(neighbors, key=lambda x: node_degree_dict.get(x, 0), reverse=True)
            for neighbor in neighbors_sorted:
                queue.append(neighbor)
    
    if current_block:
        if len(current_block) <= block_size and len(blocks) > 0:
            if len(blocks[-1]) + len(current_block) <= block_size * 1.5:
                blocks[-1].extend(current_block)
            else:
                blocks.append(current_block)
        else:
            blocks.append(current_block)
    
    unvisited_nodes = [node for node in all_nodes if node not in visited]
    if unvisited_nodes:
        for i in range(0, len(unvisited_nodes), block_size):
            small_block = unvisited_nodes[i:i+block_size]
            if small_block:
                blocks.append(small_block)
    
    return blocks, list(set([n for b in blocks for n in b]))


def split_data_random(all_nodes, block_size, seed=42):
    """Random partitioning"""
    random.seed(seed)
    nodes_shuffled = list(all_nodes)
    random.shuffle(nodes_shuffled)
    
    blocks = []
    for i in range(0, len(nodes_shuffled), block_size):
        block = nodes_shuffled[i:i+block_size]
        if block:
            blocks.append(block)
    return blocks


def process_block(args, edge_df, cn_base_alpha):
    """Process block for edge weight calculation"""
    block_id, block_nodes = args
    return calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha)


def calc_block_edge_weight_no_queue(edge_df, block_nodes, block_id, cn_base_alpha):
    """Calculate block edge weights"""
    try:
        block_node_set = set(block_nodes)
        block_mask = edge_df['u'].isin(block_node_set) & edge_df['v'].isin(block_node_set)
        block_edge = edge_df[block_mask].copy()
        if len(block_edge) == 0:
            return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))
        
        block_edge[['u_sorted', 'v_sorted']] = np.sort(block_edge[['u', 'v']].values, axis=1)
        edge_counts = block_edge.groupby(['u_sorted', 'v_sorted']).size().reset_index(name='count')
        
        neighbor_dict = defaultdict(list)
        for _, row in edge_counts.iterrows():
            u, v = row['u_sorted'], row['v_sorted']
            neighbor_dict[u].append(v)
            neighbor_dict[v].append(u)
        for u in neighbor_dict:
            neighbor_dict[u].sort()
        
        def count_common(u, v):
            neighbors_u = neighbor_dict.get(u, [])
            neighbors_v = neighbor_dict.get(v, [])
            i = j = common = 0
            len_u, len_v = len(neighbors_u), len(neighbors_v)
            while i < len_u and j < len_v:
                if neighbors_u[i] == neighbors_v[j]:
                    common += 1
                    i += 1
                    j += 1
                elif neighbors_u[i] < neighbors_v[j]:
                    i += 1
                else:
                    j += 1
            return common
        
        edge_counts['common'] = edge_counts.apply(lambda row: count_common(row['u_sorted'], row['v_sorted']), axis=1)
        edge_counts['weight'] = edge_counts['count'] + cn_base_alpha * edge_counts['common']
        
        result_edge = block_edge[['u', 'v']].drop_duplicates()
        result_edge = result_edge.merge(edge_counts[['u_sorted', 'v_sorted', 'weight']],
                                        left_on=['u', 'v'],
                                        right_on=['u_sorted', 'v_sorted'],
                                        how='left').fillna(1)[['u', 'v', 'weight']]
        return (block_id, result_edge)
    except Exception as e:
        print(f"Block {block_id} edge weight calculation failed: {str(e)}")
        return (block_id, pd.DataFrame(columns=['u', 'v', 'weight']))


# ==================== Adaptive Parameter Calculation ====================

def compute_block_size_theory(N, E):
    """Calculate block size based on theory"""
    import math
    
    # BYTES_PER_NODE = 100
    # TARGET_MEM_MB = 50
    BLOCKS_PER_CORE = 3
    num_workers = 5
    
    SPARSE_DEGREE = 1
    DENSE_DEGREE = 30
    SPARSE_FACTOR = 0
    DENSE_FACTOR = 2
    
    avg_deg = 2 * E / N
    
    if avg_deg < SPARSE_DEGREE:
        density = SPARSE_FACTOR
    elif avg_deg > DENSE_DEGREE:
        density = DENSE_FACTOR
    else:
        density = SPARSE_FACTOR + (DENSE_FACTOR - SPARSE_FACTOR) / (DENSE_DEGREE - SPARSE_DEGREE) * (avg_deg - SPARSE_DEGREE)
    
    print(f"avg_deg: {avg_deg}")
    print(f"density: {density}")
    
    # max_by_mem = (TARGET_MEM_MB * 1024 * 1024) // BYTES_PER_NODE
    max_by_parallel = N // (num_workers * BLOCKS_PER_CORE)
    block = int(max_by_parallel * density)
    block_size = max(1000, min(block, 50000))
    
    d_critical = 6
    base_tau = max(2, int(math.log10(N)))
    
    if avg_deg < d_critical:
        tau = max(2, base_tau - 1)
    else:
        tau = base_tau
    
    # alpha = 0.5 if avg_deg < 10 else 0.2
    k_min, k_max = 5, 200  # 根据 Lü et al. 的经验范围
    alpha_max, alpha_min = 2.0, 0.0
    
    if avg_deg <= k_min:
        alpha = alpha_max
    elif avg_deg >= k_max:
        alpha = alpha_min
    else:
        # 对数插值
        log_ratio = (math.log(avg_deg) - math.log(k_min)) / \
                    (math.log(k_max) - math.log(k_min))
        # 非线性调整 (γ=0.8)
        alpha = alpha_max + (alpha_min - alpha_max) * (log_ratio ** 0.8)

    # SPARSE_DEGREE = 1    # 稀疏阈值
    # DENSE_DEGREE = 30     # 稠密阈值
    # ALPHA_SPARSE = 2    # 稀疏时共同邻居权重大
    # ALPHA_DENSE = 0    # 稠密时共同邻居权重小

    # if avg_deg <= SPARSE_DEGREE:
    #     alpha = ALPHA_SPARSE      
    # elif avg_deg >= DENSE_DEGREE:
    #     alpha = ALPHA_DENSE       
    # else:
    #     slope = (ALPHA_DENSE - ALPHA_SPARSE) / (DENSE_DEGREE - SPARSE_DEGREE)  
    #     alpha = ALPHA_SPARSE + slope * (avg_deg - SPARSE_DEGREE)
        
    alpha = round(alpha, 2)
    
    return block_size, tau, alpha


def choose_chunking_strategy_theoretical(edges, total_nodes, block_size=20000, full_graph_threshold=100000, sample_size=10000, seed=42, cv_threshold=0.6):
    """Choose chunking strategy based on spectral coefficient"""
    print(f" cv_threshold = {cv_threshold:.4f}")
    kappa, lambda2, cv_cc = compute_spectral_coefficient_and_cv(edges, total_nodes, block_size, full_graph_threshold, sample_size, seed)
    
    if kappa < 2 and cv_cc < cv_threshold:
        strategy = "BFS"
        reason = f"Spectral coeff kappa={kappa:.4f} < 2 and uniform local density (cv_cc={cv_cc:.3f} < {cv_threshold}), clear community structure, BFS preserves locality"
    else:
        strategy = "Random"
        reason = f"Spectral coeff kappa={kappa:.4f} >= 2 or non-uniform density (cv_cc={cv_cc:.3f} >= {cv_threshold}), unclear boundaries, random better for load balance"
    
    
    
    return {
        "strategy": strategy,
        "kappa": kappa,
        "lambda2": lambda2,
        "cv_cc": cv_cc,
        "reason": reason,
        # "cv_degree": cv_degree
    }


def compute_spectral_coefficient_and_cv(edges, total_nodes, block_size=20000, full_graph_threshold=100000, sample_size=10000, seed=42):
    """Compute spectral coefficient"""
    if total_nodes <= block_size:
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        G.add_edges_from(edges)
        L = nx.normalized_laplacian_matrix(G).astype(np.float32)
        try:
            eigenvalues, _ = eigsh(L, k=2, which='SM', tol=1e-4)
            lambda2 = eigenvalues[1]
        except Exception:
            lambda2 = 0.0
        _, _, cv_cc = compute_clustering_coeff_stats(G)
        return 0.0, lambda2, cv_cc
    
    if total_nodes <= full_graph_threshold:
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        G.add_edges_from(edges)
        L = nx.normalized_laplacian_matrix(G).astype(np.float32)
        try:
            eigenvalues, _ = eigsh(L, k=2, which='SM', tol=1e-4)
            lambda2 = eigenvalues[1]
        except Exception:
            lambda2 = 0.0
        _, _, cv_cc = compute_clustering_coeff_stats(G)
        kappa = lambda2 * block_size / (total_nodes - block_size) if total_nodes > block_size else 1e9
        return kappa, lambda2, cv_cc
    
    G, _ = build_connected_sampled_graph(edges, total_nodes, sample_size, seed)
    L = nx.normalized_laplacian_matrix(G).astype(np.float32)
    try:
        eigenvalues, _ = eigsh(L, k=2, which='SM', tol=1e-4)
        lambda2 = eigenvalues[1]
    except Exception:
        lambda2 = 0.0
    _, _, cv_cc = compute_clustering_coeff_stats(G)
    kappa = lambda2 * block_size / (total_nodes - block_size) if total_nodes > block_size else 1e9
    return kappa, lambda2, cv_cc


def compute_clustering_coeff_stats(G):
    """Compute clustering coefficient statistics"""
    clustering = list(nx.clustering(G).values())
    if not clustering:
        return 0.0, 0.0, 0.0
    mean_cc = np.mean(clustering)
    std_cc = np.std(clustering)
    cv_cc = std_cc / mean_cc if mean_cc > 0 else 1.0
    return mean_cc, std_cc, cv_cc


def compute_degree_cv(edges, total_nodes, sample_size=10000):
    """Compute degree distribution CV"""
    if total_nodes < 100000:
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        G.add_edges_from(edges)
        degrees = [d for n, d in G.degree()]
    else:
        degree_count = Counter()
        for u, v in edges:
            degree_count[u] += 1
            degree_count[v] += 1
        sampled_nodes = np.random.choice(list(degree_count.keys()), size=min(sample_size, len(degree_count)), replace=False)
        degrees = [degree_count[n] for n in sampled_nodes]
    
    avg_deg = np.mean(degrees)
    std_deg = np.std(degrees)
    cv_degree = std_deg / avg_deg if avg_deg > 0 else 0
    return cv_degree


def build_connected_sampled_graph(edges, total_nodes, sample_size=10000, seed=42):
    """Random walk sampling"""
    np.random.seed(seed)
    random.seed(seed)
    if total_nodes <= sample_size:
        G = nx.Graph()
        G.add_nodes_from(range(total_nodes))
        G.add_edges_from(edges)
        return G, total_nodes
    
    adj = [[] for _ in range(total_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    start = np.random.randint(total_nodes)
    visited = set([start])
    queue = [start]
    while len(visited) < sample_size and queue:
        current = queue.pop(0)
        neighbors = adj[current]
        random.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= sample_size:
                    break
    
    sampled_nodes = list(visited)
    sampled_edges = [(u, v) for u, v in edges if u in visited and v in visited]
    G = nx.Graph()
    G.add_nodes_from(sampled_nodes)
    G.add_edges_from(sampled_edges)
    return G, len(sampled_nodes)


# ==================== Data Loading ====================

def load_data(edge_file_path, comm_file_path):
    """Load SNAP dataset"""
    with open(comm_file_path) as f:
        communities = [[int(i) for i in x.split()] for x in f]
    
    with open(edge_file_path) as f:
        edges = [[int(i) for i in e.split()] for e in f]
    
    edges = [[u, v] if u < v else [v, u] for u, v in edges if u != v]
    
    raw_nodes = {node for e in edges for node in e}
    mapping = {u: i for i, u in enumerate(sorted(raw_nodes))}
    
    edges = [[mapping[u], mapping[v]] for u, v in edges]
    communities = [[mapping[node] for node in com] for com in communities]

    # filtered = [s for s in communities if 7 <= len(s) ]
    # print(len(filtered))  # 尝试是否得到 599
    # print(len(filtered))  # 尝试是否得到 599
    
    num_node, num_edges, num_comm = len(raw_nodes), len(edges), len(communities)
    print(f"[{os.path.basename(edge_file_path).upper()}] #Nodes {num_node}, #Edges {num_edges}, #Communities {num_comm}")
    
    new_nodes = list(range(len(raw_nodes)))
    return num_node, num_edges, num_comm, new_nodes, edges, communities


# ==================== 数据加载（支持 Cora） ====================
def load_planetoid(data_dir, dataset):
    """
    加载 Planetoid 文本格式，通过严格对齐测试节点索引和边列表，
    使其输出与 .mat 格式完全一致。
    返回: (features, labels, all_nodes, edges, communities)
    """
    # 1. 加载原始文件
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objs = {}
    for name in names:
        with open(f"{data_dir}/{dataset}/ind.{dataset}.{name}", 'rb') as f:
            # print(f"{data_dir}/{dataset}/ind.{dataset}.{name}")
            objs[name] = pickle.load(f, encoding='latin1')
    x = objs['x']          # 训练+验证特征 (稀疏)
    y = objs['y']          # 训练+验证标签 (one-hot)
    tx = objs['tx']        # 测试特征
    ty = objs['ty']        # 测试标签
    allx = objs['allx']    # 训练+验证+部分测试？实际是训练+验证
    ally = objs['ally']
    graph = objs['graph']
    
    test_idx = np.loadtxt(f"{data_dir}/{dataset}/ind.{dataset}.test.index", dtype=int)
    
    # 2. 确定所有节点的原始ID列表（从0到最大索引）
    # 训练+验证节点的ID范围：0..len(allx)-1（假设是连续的）
    # 测试节点ID由 test_idx 给出，可能不连续
    all_ids = set(range(allx.shape[0])) | set(test_idx)
    max_id = max(all_ids)
    num_node = max_id + 1   # 3327 for Citeseer
    
    # 3. 构建特征矩阵 (num_node, dim)
    dim = allx.shape[1]
    features = sp.lil_matrix((num_node, dim), dtype=np.float32)
    # 填充训练+验证节点 (allx)
    features[:allx.shape[0]] = allx
    # 填充测试节点 (tx) 到对应索引
    for i, idx in enumerate(test_idx):
        features[idx] = tx[i]
    features = features.tocsr()
    
    # 4. 构建标签向量 (num_node,)
    labels = np.zeros(num_node, dtype=np.int64)
    # 训练+验证标签 (ally) 是 one-hot，转为整数
    train_labels = np.argmax(ally, axis=1)
    labels[:len(train_labels)] = train_labels
    # 测试标签
    test_labels = np.argmax(ty, axis=1)
    for i, idx in enumerate(test_idx):
        labels[idx] = test_labels[i]
    
    # 5. 构建边列表（与 .mat 的 W 矩阵一致）
    # .mat 中的 W 通常是对称的，且包含自环？需要确认。一般 Planetoid 的 graph 无自环。
    # 为了对齐，我们构建无自环的无向图，每条边只存储一次 (i < j)
    G = nx.from_dict_of_lists(graph)
    # 确保节点范围包含所有节点（可能 graph 中缺少孤立节点）
    G.add_nodes_from(range(num_node))
    # 导出边
    edge_set = set()
    edges = []
    for u, v in G.edges():
        if u == v: continue
        u, v = sorted((u, v))
        if (u, v) not in edge_set:
            edge_set.add((u, v))
            edges.append([u, v])
    # 注意：.mat 中的 W 可能比这个边数多（比如包含自环），但通常 Citeseer 无自环。
    # 为了完全一致，可以检查 .mat 加载的边数，如果多了就说明 .mat 有重复或方向。
    # 但上述边构建应该与 .mat 的 `i<j` 过滤结果一致。
    
    # 6. 构建社区（按标签分组）
    label_to_nodes = defaultdict(list)
    for node, lab in enumerate(labels):
        label_to_nodes[lab].append(node)
    communities = [label_to_nodes[i] for i in sorted(label_to_nodes.keys())]
    all_nodes = list(range(num_node))
    
    # 7. 可选：特征转换为密集（如果 .mat 是密集的话）
    # 为了与 .mat 完全一致，需要检查 .mat 中 fea 是稀疏还是密集。通常 .mat 的 fea 是稀疏，但加载后可能转为密集。
    # 这里保持稀疏，但后续使用时可以 .toarray()
    features = features.toarray().astype(np.float32)
    
    print(f"Aligned {dataset}: nodes={num_node}, edges={len(edges)}, communities={len(communities)}, feature_dim={features.shape[1]}")
    return features, labels, all_nodes, edges, communities




# def load_cora(data_dir, dataset_name='cora'):
#     """专门加载 Cora 数据集，返回统一格式"""
#     features, labels, all_nodes, edges, communities = load_planetoid(data_dir, dataset_name)
#     num_node = len(all_nodes)
#     num_edges = len(edges)
#     num_comm = len(communities)
#     return num_node, num_edges, num_comm, all_nodes, edges, communities, features

def evaluate_with_correct_format(true_comms, comm_dict):
    """Evaluation function"""
    from collections import defaultdict
    
    pred_comms = defaultdict(list)
    for node, comm_id in comm_dict.items():
        pred_comms[comm_id].append(node)
    pred_comms = list(pred_comms.values())
    
    try:
        from metrics import eval_scores_fast_optimized_fixed
        avg_precision, avg_recall, avg_f1, avg_jaccard = eval_scores_fast_optimized_fixed(pred_comms, true_comms, tmp_print=True)
        print(f"  Average Precision: {avg_precision:.4f}")
        print(f"  Average Recall: {avg_recall:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        print(f"  Average Jaccard: {avg_jaccard:.4f}")
        return avg_precision, avg_recall, avg_f1, avg_jaccard
    except ImportError:
        print("Warning: metrics module not found, using fallback evaluation")
        return 0.0, 0.0, 0.0, 0.0

def load_cora(root='./data/', dataset='cora'):
    import scipy.io as sio
    file_path = os.path.join(root, '{}.mat'.format(dataset))
    data = sio.loadmat(file_path)
    if dataset in ["BlogCatalog", "Flickr"]:
        feature = data['Attributes']
        adj = data['Network']
        gnd = data['Label']
    else:
        feature = data['fea']
        adj = data['W']
        gnd = data['gnd']
    if sp.issparse(feature):
        feature = feature.todense()
    gnd = gnd.T - 1
    gnd = gnd[0, :]
    num_node = feature.shape[0]
    dim_features = feature.shape[1]
    num_comm = len(np.unique(gnd))
    if not sp.issparse(adj):
        adj = sp.csr_matrix(adj)
    adj_coo = adj.tocoo()
    rows, cols = adj_coo.row, adj_coo.col
    edges = []
    for i, j in zip(rows, cols):
        if i < j:
            edges.append([i, j])
    num_undirected_edges = len(edges)
    # num_edges = num_undirected_edges * 2
    num_edges = num_undirected_edges
    all_nodes = list(range(num_node))
    label_to_nodes = defaultdict(list)
    for node_idx, label in enumerate(gnd):
        label_to_nodes[label].append(node_idx)
    communities = [label_to_nodes[i] for i in range(num_comm)]
    features_tensor = torch.tensor(feature, dtype=torch.float32)
    print(f"[{dataset}] loaded: nodes={num_node}, edges={num_edges}, features={dim_features}, classes={num_comm}")
    return num_node, num_edges, num_comm, all_nodes, edges, communities, features_tensor


# ==================== Main Pipeline ====================
def gen_nci(F, k, max_iter=50, tol=1e-5):
    n = F.shape[1]
    device = F.device
    Y = torch.zeros(k, n, dtype=torch.float32, device=device)
    for j in range(n):
        idx = torch.randint(0, k, (1,)).item()
        Y[idx, j] = 1.0
    X = torch.eye(k, device=device)
    for it in range(max_iter):
        M = F.T @ X.T
        gamma = torch.sqrt(Y.sum(dim=1))
        for j in range(n):
            best_c = -1
            best_val = -float('inf')
            for c in range(k):
                Y_cj = Y[c, j].item()
                gamma_c = gamma[c].item()
                term = ((1 - Y_cj) * M[j, c].item()) / np.sqrt(gamma_c**2 + 1) + \
                       (Y_cj * M[j, c].item()) / max(gamma_c, 1e-8)
                if term > best_val:
                    best_val = term
                    best_c = c
            Y[:, j] = 0.0
            Y[best_c, j] = 1.0
            gamma[best_c] = torch.sqrt(Y[best_c].sum())
        YYt = Y @ Y.T
        try:
            sqrt_inv = torch.linalg.pinv(torch.linalg.cholesky(YYt + 1e-8 * torch.eye(k, device=device)))
        except:
            sqrt_inv = torch.linalg.pinv(torch.sqrt(YYt + 1e-8 * torch.eye(k, device=device)))
        H = sqrt_inv @ Y
        U, S, Vt = torch.linalg.svd(H @ F.T, full_matrices=False)
        X_new = U @ Vt
        diff = torch.norm(X_new - X, p='fro')
        X = X_new
        if diff < tol:
            break
    return Y

def cluster_with_method(embed_matrix, k, method):
    if method == 'kmeans':
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        return kmeans.fit_predict(embed_matrix)
    # else:  # gennci
    #     from scipy.sparse.linalg import svds
    #     U, S, Vt = svds(embed_matrix, k=k)
    #     F_tensor = torch.tensor(np.ascontiguousarray(U.T), dtype=torch.float32)
    #     Y = gen_nci(F_tensor, k=k, max_iter=50)
    #     return torch.argmax(Y, dim=0).cpu().numpy()

def compute_avg_degree(edges, num_nodes):
    degree = Counter()
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    return np.mean(list(degree.values()))

def auto_select_encoder(edge_index, x, original_feat_dim, num_epochs=50, device='cpu', verbose=True):
    """
    Automatically select better encoder (GCN vs GAT) based on validation silhouette,
    with a fallback rule: if original_feat_dim < 1000, always choose GCN.
    """
    n_nodes = x.shape[0]
    val_size = max(100, int(n_nodes * 0.2))
    val_idx = torch.randperm(n_nodes, device=device)[:val_size]
    scores = {}
    
    # 低维特征直接返回 GCN，无需预训练
    if original_feat_dim < 1000:
        if verbose:
            print(f"Low-dimensional features ({original_feat_dim} < 1000) -> forcing GCN")
        return 'gcn'
    
    for enc in ['gcn', 'gat']:
        if enc == 'gcn':
            model = GCN_Encoder(x.shape[1], HIDDEN_DIM).to(device)
        else:
            model = GAT_Encoder(x.shape[1], HIDDEN_DIM).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        edge_index_dev = edge_index.to(device)
        x_dev = x.to(device)
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            z = model(x_dev, edge_index_dev)
            adj_pred = torch.sigmoid(torch.mm(z, z.t()))
            pos_loss = -torch.log(adj_pred[edge_index_dev[0], edge_index_dev[1]] + 1e-8).mean()
            neg_indices = torch.randint(0, n_nodes, (2, edge_index_dev.size(1)), device=device)
            neg_loss = -torch.log(1 - adj_pred[neg_indices[0], neg_indices[1]] + 1e-8).mean()
            loss = pos_loss + neg_loss
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            z_full = model(x_dev, edge_index_dev)
            z_val = z_full[val_idx].cpu().numpy()
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        k = min(15, val_size-1)
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=10)
        labels = kmeans.fit_predict(z_val)
        sil = silhouette_score(z_val, labels) if len(set(labels)) > 1 else -1
        scores[enc] = sil
        if verbose:
            print(f"Quick eval: {enc.upper()} silhouette={sil:.4f}")
    
    best = max(scores, key=scores.get)
    if verbose:
        print(f"Auto-selected encoder: {best.upper()} (silhouette diff: {scores['gat']-scores['gcn']:.4f})")
    return best

def decide_strategy(N, num_classes, avg_deg):
    # if avg_deg > 6 and num_classes > 500:
    #     return 'both'
    # elif N<10000 and 
    if N<5000 and avg_deg <= 6:
        return 'emb'
    else:
        return 'feat'

# def decide_strategy_by_diffusion(G, feat_matrix):
#     """
#     基于一步扩散的变化率自动决策。
#     返回 'feat' 或 'both'
#     """
#     from scipy.sparse.linalg import eigsh
#     import scipy.sparse as sp
    
#     N = feat_matrix.shape[0]
#     # 构建归一化拉普拉斯矩阵
#     adj = nx.adjacency_matrix(G)
#     deg = np.array(adj.sum(axis=1)).flatten()
#     deg_inv_sqrt = np.power(deg + 1e-8, -0.5)
#     D_inv_sqrt = sp.diags(deg_inv_sqrt)
#     L = sp.eye(N) - D_inv_sqrt @ adj @ D_inv_sqrt  # 归一化拉普拉斯
    
#     # 一步扩散
#     alpha = 0.5
#     X = feat_matrix
#     X_prime = X - alpha * (L @ X)
#     delta = np.linalg.norm(X_prime - X) / np.linalg.norm(X)
#     print(f"delta:{delta}")
    
#     # 阈值可通过实验确定（此处取 0.3，可根据您的数据调整）
#     if delta > 0.3:
#         return 'both'
#     else:
#         return 'feat'



from scipy.sparse.linalg import eigsh

def compute_adaptive_iterations(norm_adj, N, C, alpha=0.15, min_iters=100, max_iters=500):
    """
    基于谱混合时间理论，完全由数据驱动计算最大迭代轮数。
    无任何经验常数。
    """
    try:
        # 计算归一化邻接矩阵的前两大特征值（'LM' 取模最大的）
        # 注意：最大特征值通常为 1.0，第二大特征值 λ₂ 编码了社区结构
        eigenvalues = eigsh(norm_adj, k=2, which='LM', tol=1e-3, return_eigenvectors=False)
        eigenvalues = np.sort(eigenvalues)[::-1]  # 降序排列
        lambda2 = eigenvalues[1]  # 第二大特征值
        
        # 谱隙（社区强度的度量）
        spectral_gap = 1 - lambda2
        if spectral_gap < 1e-6:
            spectral_gap = 1e-6  # 安全保护
        
        # 理论公式：T = ln(N/C) / (alpha * spectral_gap)
        s = N / max(C, 1)
        adaptive_iters = int(np.log(s) / (alpha * spectral_gap))
        adaptive_iters = max(min_iters, min(max_iters, adaptive_iters))
        
        print(f"  λ₂(Ã) = {lambda2:.4f}, spectral_gap = {spectral_gap:.4f}, adaptive T = {adaptive_iters}")
        return adaptive_iters
    except Exception as e:
        # 特征值计算失败（如图过大或稀疏性问题），安全回退至 200
        print(f"  Spectral computation fallback to 200 due to: {e}")
        return 200
    
    

def  execute_HIDC_pipeline_unsupervised(edge_file_path, comm_file_path, network_type, is_cora=False, cora_data_dir=None, cora_dataset='cora'):
    """Main pipeline - improved version with structural-first path"""
    # epolist=[100,200,400,600,800]
    # dimlist=[16,32,64,128,256]
    # epolist=[600]
    # for dim in dimlist:
        # global HIDDEN_DIM
        # HIDDEN_DIM = dim

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
        torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"MESSAGE_PASSING_ITERATIONS:{MESSAGE_PASSING_ITERATIONS}")
    
    start_total_time1 = time.time()
    
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR, exist_ok=True)
    fea_num=0
    print("\n[1/6 Loading data]")
    # use_hacd_flag=True
    if is_cora:

        num_node, num_edges, num_comm, all_nodes, edges, communities, features = load_cora("dataset", dataset_name)
        
        edge_df = pd.DataFrame(edges, columns=['u', 'v'])
        print(f"  Original feature dimension: {features.shape[1]}")
        original_feat_dim=features.shape[1]
        features = normalize(features, norm='l2')
        pca = PCA(n_components=50, random_state=RANDOM_SEED)
        features = pca.fit_transform(features)
        print(f"  After PCA: {features.shape}")
        node_feat_dict = {node: features[i] for i, node in enumerate(all_nodes)}

    
    else:
        num_node, num_edges, num_comm, all_nodes, edges, communities = load_data(edge_file_path, comm_file_path)
        edge_df = pd.DataFrame(edges, columns=['u', 'v'])
        node_feat_dict = None
    
    print("\n[2/6 Building graph and structural features]")
    full_adj = defaultdict(list)
    for u, v in edges:
        full_adj[u].append(v)
        full_adj[v].append(u)
    node_degree_dict = {node: len(full_adj[node]) for node in all_nodes}
    
    print("  Computing clustering coefficients and PageRank...")
    G_full = nx.Graph()
    G_full.add_edges_from(edges)
    clustering = nx.clustering(G_full)
    pagerank = nx.pagerank(G_full)
    

    if node_feat_dict is None:
        print(f"  Generating structural features for non-attribute graph...")
        node_feat_dict = {}
        for node in all_nodes:
            # deg = node_degree_dict.get(node, 0)
            # clust_coef = clustering.get(node, 0)
            # pr = pagerank.get(node, 0)
            h = deterministic_hash(node)
            rng = np.random.RandomState(seed=(h % 10000))
            padding = rng.normal(0, 0.1, 50).astype(np.float32)
            feat = np.concatenate([padding])
            node_feat_dict[node] = feat
        print(f"  Generated structural features: dim={len(next(iter(node_feat_dict.values())))}")


        # #one-hot
        # from sklearn.preprocessing import OneHotEncoder

        # # 假设 node_degree_dict 已存在，格式为 {node: degree}
        # # 1. 获取所有节点的度，并重塑为 sklearn 需要的格式
        # degrees = np.array([node_degree_dict.get(node, 0) for node in all_nodes]).reshape(-1, 1)

        # # 2. 训练 OneHotEncoder
        # enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # enc.fit(degrees)

        # # 3. 为每个节点生成 one-hot 特征
        # one_hot_features = enc.transform(degrees)

        # # 4. 填充到 node_feat_dict
        # node_feat_dict = {}
        # for i, node in enumerate(all_nodes):
        #     # 这里直接使用 one-hot 特征，你也可以选择拼接其他特征
        #     node_feat_dict[node] = one_hot_features[i].astype(np.float32)

        # One-hot编码（与随机特征公平比较）
        # from sklearn.preprocessing import OneHotEncoder
        # from sklearn.decomposition import PCA

        # # 1. 获取所有节点的度
        # degrees = np.array([node_degree_dict.get(node, 0) for node in all_nodes]).reshape(-1, 1)

        # # 2. One-hot编码
        # enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        # one_hot_features_raw = enc.fit_transform(degrees)

        # print(f"  Raw one-hot dimension: {one_hot_features_raw.shape[1]}")

        # # 3. 降维到50维（与随机特征相同）
        # if one_hot_features_raw.shape[1] > 50:
        #     # 使用PCA降维
        #     pca = PCA(n_components=50)
        #     one_hot_features = pca.fit_transform(one_hot_features_raw)
        #     print(f"  Reduced to 50 dimensions using PCA (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
        # elif one_hot_features_raw.shape[1] < 50:
        #     # 如果维度不足50，用零填充
        #     one_hot_features = np.zeros((len(all_nodes), 50), dtype=np.float32)
        #     one_hot_features[:, :one_hot_features_raw.shape[1]] = one_hot_features_raw
        #     print(f"  Padded to 50 dimensions with zeros")
        # else:
        #     one_hot_features = one_hot_features_raw
        #     print(f"  Already 50 dimensions")

        # # 4. 填充到 node_feat_dict（确保与随机特征维度相同）
        # node_feat_dict = {}
        # for i, node in enumerate(all_nodes):
        #     node_feat_dict[node] = one_hot_features[i].astype(np.float32)

        # print(f"  Generated one-hot features: dim={len(next(iter(node_feat_dict.values())))}")

    
    N = len(all_nodes)
    C = len(communities)


    

    avg_deg = compute_avg_degree(edges, num_node)
    print(f"Average degree: {avg_deg:.2f}")

    strategy=decide_strategy(N,C,avg_deg)


    print(f"strategy:{strategy}")


    assortativity = nx.degree_assortativity_coefficient(G_full)
    print(f"Assortativity: {assortativity:.4f}")
    use_cc=True
    if assortativity>0.4 and  avg_deg>6 :
        use_cc=False

    # 软门控方案：计算门控权重，平滑过渡GCN和GAT
    # gate_weight = compute_gate_weight_smooth(assortativity, avg_deg, R_low=-0.05, R_high=0.05, D_thresh=D)
    gate_weight = compute_gate_weight_smooth(assortativity, avg_deg, R_low=-0.05, R_high=0.05, D_thresh=D)
    print(f"Soft gating weight: {gate_weight:.4f} (0=pure GCN, 1=pure GAT)")
    
    # 根据门控权重决定是否归一化（偏向GAT时不归一化，偏向GCN时归一化）
    need_nom = gate_weight < 0.5 and avg_deg < 6
    # encoder = 'gat' if gate_weight > 0.5 else 'gcn'  # 保留变量兼容其他地方引用
    cluster_method = 'kmeans'


        # 3. 选择聚类方法
    # if avg_deg > 10:
    #     cluster_method = 'gennci'
    # else:
    #     cluster_method = 'kmeans'
    # print(f"Auto-selected clustering method: {cluster_method.upper()}")

        # # 选择编码器和聚类方法
    # if avg_deg > 30:
    #     encoder = 'gcn'
    #     cluster_method = 'gennci'
    # else:
    #     if assortativity < -0.05 and avg_deg < 6 :
    #         encoder = 'gat'           
    #     else:
    #         encoder = 'gcn'
    #     cluster_method = 'kmeans'
    # print(f"cluster_method: {cluster_method},encoder:{encoder}")

    
    # ========== Small Graph Mode ==========
    if avg_deg < 10 and N <= cut_blocknum:
    # if  N <= 20000:

        # if avg_deg < 10 and fea_num > 1000:
        #     encoder = 'gat'
        # else:
        #     encoder = 'gcn'

        print(f"\n[3/6] Small graph ({N} nodes, {C} communities) -> full-graph HACD")
        node_list = all_nodes
        feat_matrix = np.array([node_feat_dict[node] for node in node_list], dtype=np.float32)
        x_tensor = torch.tensor(feat_matrix, dtype=torch.float32)
        edge_index_tensor = torch.tensor([[u, v] for u, v in edges], dtype=torch.long).t().contiguous()
        edge_index_tensor = to_undirected(edge_index_tensor)

         # alpha = 0.5 if avg_deg < 10 else 0.2
        k_min, k_max = 5, 200  # 根据 Lü et al. 的经验范围
        alpha_max, alpha_min = 2.0, 0.0
        
        if avg_deg <= k_min:
            alpha = alpha_max
        elif avg_deg >= k_max:
            alpha = alpha_min
        else:
            # 对数插值
            log_ratio = (math.log(avg_deg) - math.log(k_min)) / \
                        (math.log(k_max) - math.log(k_min))
            # 非线性调整 (γ=0.8)
            alpha = alpha_max + (alpha_min - alpha_max) * (log_ratio ** 0.8)
        
        cn_base_alpha = round(alpha, 2)
        # cn_base_alpha=0


        if gate_weight>=1:
           cn_base_alpha=0

        # 计算边权重（使 cn_base_alpha 生效）
        print(f"  Computing edge weights (cn_base_alpha={cn_base_alpha})...")
        _, weighted_edges = calc_block_edge_weight_no_queue(edge_df, all_nodes, 0, cn_base_alpha)
        # if not weighted_edges.empty:
        #     # 构建边权重张量
        #     edge_weight_map = dict(zip(zip(weighted_edges['u'], weighted_edges['v']), weighted_edges['weight']))
        #     edge_list = edge_index_tensor.t().tolist()
        #     edge_weights = [edge_weight_map.get((u, v), edge_weight_map.get((v, u), 1.0)) for u, v in edge_list]
        #     edge_weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
        # else:
        #     edge_weight_tensor = None
            

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        hacd_emb = get_hacd_embeddings_soft_gate(edge_index_tensor, x_tensor, num_epochs=EPOCHS, lr=0.01, device=device, hidden_dim=HIDDEN_DIM, gate_weight=gate_weight)
        
        deg = np.log1p([node_degree_dict[node] for node in node_list]).reshape(-1,1)
        clust = np.array([clustering.get(node, 0) for node in node_list]).reshape(-1,1)
        pr_vals = np.array([pagerank.get(node, 0) for node in node_list]).reshape(-1,1)

        if not weighted_edges.empty:
            # 聚合 u 和 v 方向（假设边是单向存储，无向图每条边仅出现一次）
            wdeg_u = weighted_edges.groupby('u')['weight'].sum().to_dict()
            wdeg_v = weighted_edges.groupby('v')['weight'].sum().to_dict()
            wdeg_dict = {}
            for k, v in wdeg_u.items():
                wdeg_dict[k] = wdeg_dict.get(k, 0) + v
            for k, v in wdeg_v.items():
                wdeg_dict[k] = wdeg_dict.get(k, 0) + v
        else:
            wdeg_dict = {}
        
        weighted_deg = np.log1p([wdeg_dict.get(node, 0) for node in all_nodes]).reshape(-1, 1)


        if use_cc:
            if strategy=="emb":
                combined = np.hstack([hacd_emb, weighted_deg, clust])
            elif strategy=="feat":
                combined = np.hstack([feat_matrix, weighted_deg, clust])
        else:
            if strategy=="emb":
                combined = np.hstack([hacd_emb])
            elif strategy=="feat":
                combined = np.hstack([feat_matrix])

        # else:
        #     combined = np.hstack([hacd_emb,feat_matrix, deg, clust])
        # combined = np.hstack([feat_matrix, deg, clust])

        if need_nom:
            combined = normalize(combined, norm='l2')
        # combined = normalize(combined, norm='l2')
        
        if combined.shape[1] > HIDDEN_DIM:
            pca2 = PCA(n_components=HIDDEN_DIM, random_state=RANDOM_SEED)
            combined = pca2.fit_transform(combined)
        elif combined.shape[1] < HIDDEN_DIM:
            combined = np.pad(combined, ((0,0), (0, HIDDEN_DIM - combined.shape[1])), mode='constant')
        
        # global_embed = {node: combined[i] for i, node in enumerate(node_list)}
        k = C
        # kmeans = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=20)
        # pred_labels = kmeans.fit_predict(combined)
        pred_labels = cluster_with_method(combined, k, cluster_method)
        
        true_node_to_comm = {node: cid for cid, nodes in enumerate(communities) for node in nodes}
        eval_nodes = [node for node in node_list if node in true_node_to_comm]
        if len(eval_nodes) > 0:
            true_labels = [true_node_to_comm[node] for node in eval_nodes]
            pred_labels_eval = [pred_labels[node_list.index(node)] for node in eval_nodes]
            acc, f1 = cluster_acc(np.array(true_labels), np.array(pred_labels_eval))
            nmi = nmi_score(true_labels, pred_labels_eval)
            ari = ari_score(true_labels, pred_labels_eval)
            print(f"\n[Full-graph HACD+KMeans] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f}")
        
        final_comm_dict = {node: pred_labels[i] for i, node in enumerate(node_list)}
        # 2. 计算模块度
        # Convert per-node labels to community-set format (list of sets/frozensets)
        comm_to_nodes = {}
        for node, cid in zip(node_list, pred_labels):
            cid = int(cid)
            comm_to_nodes.setdefault(cid, set()).add(node)
        communities_for_mod = list(comm_to_nodes.values())
        # Ensure G_full has all nodes and partition covers all nodes exactly once
        try:
            # Add any missing nodes to G_full (isolated nodes not in edges)
            G_full_mod = G_full.copy()
            for node in node_list:
                if node not in G_full_mod:
                    G_full_mod.add_node(node)
            # 验证 partition 覆盖所有节点
            mod = nx_comm.modularity(G_full_mod, communities_for_mod)
            print(f"模块度: {mod}")
        except Exception as e:
            print(f"模块度=计算失败: {e}")

        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        total_time = (time.time() - start_total_time1) / 60
        print(f"\nTotal time: {total_time:.2f} minutes")
        # return final_comm_dict
    else:
    
       
        # ========== Attribute Graph Path (Cora, Citeseer, PubMed):Embeddings ==========
        print(f"\n[3/6] Large attribute graph ({N} nodes, {C} communities) -> block-wise HACD")
        
        sample_size = min(10000, num_node)
        # cv_degree = compute_degree_cv(edges, total_nodes)
        # print(f"Degree distribution: CV={cv_degree:.2f}")
        block_size, MIN_COMM_SIZE, cn_base_alpha = compute_block_size_theory(len(all_nodes), len(edges))
        print(f"cn_base_alpha:{cn_base_alpha}")
        # # block_size = 3000
        # cn_base_alpha=0
        print(f"cn_base_alpha:{cn_base_alpha}")
        print(f"块大小:{block_size}")
        result = choose_chunking_strategy_theoretical(edges, num_node, block_size=block_size, sample_size=sample_size, seed=42)
        print(f"Recommended strategy: {result['strategy']}")
        print(f"Kappa = {result['kappa']:.4f}, Lambda2 = {result['lambda2']:.4f}")
        print(f"Reason: {result['reason']}")
        
        lambda2 = result['lambda2']
        cv_deg = result['cv_cc']
        

        
        if result['strategy'] == 'BFS':
            print("  Partitioning: BFS")
            base_blocks, new_all_nodes = split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size)
        else:
            print("  Partitioning: Random")
            base_blocks = split_data_random(all_nodes, block_size)
        # base_blocks, new_all_nodes = split_data_by_connectivity(edge_df, all_nodes, node_degree_dict, block_size)
        overlap_ratio = OVERLAP_RATIO
        olist = [block_size]
        
        for block_size in olist:

            overlapping_blocks = []
            for block in base_blocks:
                block_set = set(block)
                neighbor_dict = defaultdict(int)
                for node in block:
                    for nbr in full_adj.get(node, []):
                        if nbr not in block_set:
                            neighbor_dict[nbr] += 1
                neighbor_list = sorted(neighbor_dict.keys(), key=lambda n: neighbor_dict[n] * node_degree_dict.get(n, 1), reverse=True)
                overlap_size = int(len(block) * overlap_ratio)
                extra = neighbor_list[:overlap_size]
                overlapping_blocks.append(list(block_set | set(extra)))
            
            filtered_blocks = []
            small_nodes = []
            for blk in overlapping_blocks:
                if len(blk) >= MIN_BLOCK_SIZE:
                    filtered_blocks.append(blk)
                else:
                    small_nodes.extend(blk)
            if small_nodes and filtered_blocks:
                per_block = max(1, len(small_nodes) // len(filtered_blocks))
                for i in range(0, len(small_nodes), per_block):
                    chunk = small_nodes[i:i+per_block]
                    idx = i // per_block if i // per_block < len(filtered_blocks) else -1
                    if idx >= 0:
                        filtered_blocks[idx].extend(chunk)
                    else:
                        filtered_blocks[-1].extend(chunk)
            blocks = filtered_blocks
            print(f"  Overlapping blocks: {len(blocks)} blocks, avg size={np.mean([len(b) for b in blocks]):.0f}")

             # Edge weight calculation (保持不变)
            with mp.Pool(processes=NUM_PROCESSES, initializer=init_worker, initargs=(RANDOM_SEED,)) as pool:
                block_args = [(bid, block_nodes) for bid, block_nodes in enumerate(blocks)]
                partial_func = partial(process_block, edge_df=edge_df, cn_base_alpha=cn_base_alpha)
                results = pool.imap_unordered(partial_func, block_args)
                weighted_edge_dict = {bid: bedge for bid, bedge in results}

            weighted_edge_list = [weighted_edge_dict[i] for i in sorted(weighted_edge_dict.keys()) if not weighted_edge_dict[i].empty]
            if weighted_edge_list:
                weighted_edge_df = pd.concat(weighted_edge_list, ignore_index=True)
                weighted_edge_df = weighted_edge_df.astype({'u': int, 'v': int, 'weight': float})
            else:
                weighted_edge_df = pd.DataFrame(columns=['u', 'v', 'weight'])

            # Parallel block training (保持不变)
            print(f"  Training {len(blocks)} blocks in parallel...")
            block_args_list = [
                (bid, block_nodes, weighted_edge_df, node_feat_dict, node_degree_dict, clustering, pagerank, HIDDEN_DIM, RANDOM_SEED, strategy, need_nom,use_cc)
                for bid, block_nodes in enumerate(blocks)
            ]
            with mp.Pool(processes=min(NUM_PROCESSES, len(blocks)), initializer=init_worker, initargs=(RANDOM_SEED,)) as pool:
                results = pool.map(train_single_block, block_args_list)
            block_embeddings = {bid: emb for bid, emb in results}
            
            # Embedding alignment
            print("  Aligning blocks with weighted Procrustes...")
            ref_block_id = max(range(len(blocks)), key=lambda i: len(blocks[i]))
            ref_emb = block_embeddings[ref_block_id]
            aligned_emb = {node: ref_emb[node].copy() for node in ref_emb}
            
            for block_id in range(len(blocks)):
                if block_id == ref_block_id:
                    continue
                curr_emb = block_embeddings[block_id]
                common = set(ref_emb.keys()) & set(curr_emb.keys())
                if len(common) < 5:
                    aligned_emb.update(curr_emb)
                    continue
                X, Y, w = [], [], []
                for node in common:
                    w.append(np.sqrt(node_degree_dict.get(node, 1) + 1))
                    X.append(curr_emb[node])
                    Y.append(ref_emb[node])
                X = np.array(X); Y = np.array(Y); w = np.array(w)
                w = w / (w.max() + 1e-8)
                weighted_X = X * w[:, np.newaxis]
                weighted_Y = Y * w[:, np.newaxis]
                R, _ = orthogonal_procrustes(weighted_X, weighted_Y)
                for node, emb in curr_emb.items():
                    aligned_emb[node] = np.dot(emb, R)



            #6.23新
            def global_message_passing(embed_dict, adj, iterations=20, gamma=0.15, tol=1e-5):
                """
                保守优化版：与原函数逻辑完全一致，输出结果相同
                仅移除无用的prev_emb拷贝，提升约10%~15%速度
                """
                node_list = list(embed_dict.keys())
                node_to_idx = {node: i for i, node in enumerate(node_list)}
                n = len(node_list)
                emb = np.array([embed_dict[node] for node in node_list], dtype=np.float64)

                # ===== 邻接矩阵构建：与原代码完全一致 =====
                rows, cols = [], []
                for node, nbrs in adj.items():
                    i = node_to_idx.get(node)
                    if i is None:
                        continue
                    for nbr in nbrs:
                        j = node_to_idx.get(nbr)
                        if j is not None:
                            rows.append(i)
                            cols.append(j)
                
                adj_mat = sp.csr_matrix(
                    (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(n, n))

                # ===== 度与归一化：与原代码完全一致 =====
                deg = adj_mat.sum(axis=1).A1
                deg_inv = 1.0 / (deg + 1e-8)
                deg_inv_sqrt = np.sqrt(deg_inv)
                D_inv_sqrt = sp.diags(deg_inv_sqrt)
                norm_adj = D_inv_sqrt @ adj_mat @ D_inv_sqrt

                # ===== 迭代传播：公式与原代码完全一致 =====
                converged_at = iterations
                for it in range(iterations):
                    neighbor_avg = norm_adj @ emb
                    emb_new = (1 - gamma) * emb + gamma * neighbor_avg  # 原公式，确保数值一致
                    
                    norms = np.linalg.norm(emb_new, axis=1, keepdims=True)
                    emb_new = emb_new / (norms + 1e-8)
                    
                    delta = np.max(np.linalg.norm(emb_new - emb, axis=1))
                    emb = emb_new
                    
                    if delta < tol and it >= 10:
                        converged_at = it + 1
                        break

                if converged_at < iterations:
                    print(f" MP converged at iteration {converged_at}/{iterations} (delta={delta:.2e})")

                return {node: emb[i].astype(np.float32) for i, node in enumerate(node_list)}

            
            gamma=0.15
            adaptive_iters = int((1 / gamma) * avg_deg * math.log(N / max(C, 1)))
            adaptive_iters = max(50, min(500, adaptive_iters))
            print(f"  Applying global message passing (adaptive_iters {adaptive_iters} iterations)...")

            # adaptive_iters=MESSAGE_PASSING_ITERATIONS



            aligned_emb = global_message_passing(aligned_emb, full_adj, iterations=adaptive_iters, gamma=gamma)
            
            # Final KMeans
            node_list = all_nodes
            embed_matrix = np.array([aligned_emb[node] for node in node_list])
            embed_matrix = normalize(embed_matrix, norm='l2')
            k = C

            pred_labels = cluster_with_method(embed_matrix, k, cluster_method)
            
            true_node_to_comm = {node: cid for cid, nodes in enumerate(communities) for node in nodes}
            eval_nodes = [node for node in node_list if node in true_node_to_comm]
            if len(eval_nodes) > 0:
                true_labels = [true_node_to_comm[node] for node in eval_nodes]
                pred_labels_eval = [pred_labels[node_list.index(node)] for node in eval_nodes]
                acc, f1 = cluster_acc(np.array(true_labels), np.array(pred_labels_eval))
                nmi = nmi_score(true_labels, pred_labels_eval)
                ari = ari_score(true_labels, pred_labels_eval)
                print(f"\n[Block-wise HACD+KMeans] ACC={acc:.4f} NMI={nmi:.4f} ARI={ari:.4f} F1={f1:.4f}")
            
            final_comm_dict = {node: pred_labels[i] for i, node in enumerate(node_list)}

            comm_to_nodes = {}
            for node, cid in zip(node_list, pred_labels):
                cid = int(cid)
                comm_to_nodes.setdefault(cid, set()).add(node)
            communities_for_mod = list(comm_to_nodes.values())
            try:
                G_full_mod = G_full.copy()
                for node in node_list:
                    if node not in G_full_mod:
                        G_full_mod.add_node(node)
                mod = nx_comm.modularity(G_full_mod, communities_for_mod)
                print(f"模块度: {mod}")
            except Exception as e:
                print(f"模块度计算失败: {e}")
                    
        
        print("\n[Performance evaluation]")
        if communities:
            evaluate_with_correct_format(communities, final_comm_dict)
        
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
        total_time = (time.time() - start_total_time1) / 60
        print(f"\nTotal time: {total_time:.2f} minutes")
    return final_comm_dict


# ==================== Dataset Configs ====================

DATASET_CONFIGS = {

    'cora': {
        'description': 'Cora citation network',
        'network_type': 'citation',
        'is_cora': True,
        'cora_data_dir': 'dataset/cora',
        'cora_dataset': 'cora'
    },
    'citeseer': {
        'description': 'CiteSeer citation network',
        'network_type': 'citation',
        'is_cora': True,
        'cora_data_dir': 'dataset/citeseer',
        'cora_dataset': 'citeseer'
    },
    'pubmed': {
        'description': 'PubMed citation network',
        'network_type': 'citation',
        'is_cora': True,
        'cora_data_dir': 'dataset/pubmed',
        'cora_dataset': 'pubmed'
    },
    'amazon': {
        'edge_path': 'dataset/amazon/amazon-ungraph.txt',
        'community_path': 'dataset/amazon/amazon-cmty.txt',
        'description': 'Amazon co-purchasing network',
        'network_type': 'co-purchase',
        'is_cora': False
    },
    'dblp': {
        'edge_path': 'dataset/dblp/dblp-ungraph.txt',
        'community_path': 'dataset/dblp/dblp-cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration',
        'is_cora': False
    },
    'lj1': {
        'edge_path': 'dataset/lj-1.90.ungraph.txt',
        'community_path': 'dataset/lj-1.90.cmty.txt',
        'description': 'LiveJournal social network',
        'network_type': 'social',
        'is_cora': False
    },
    'dblp1': {
        'edge_path': 'dataset/dblp-1.90.ungraph.txt',
        'community_path': 'dataset/dblp-1.90.cmty.txt',
        'description': 'DBLP collaboration network',
        'network_type': 'collaboration',
        
    }

}


if __name__ == "__main__":
    import sys
    
    # Default: run non-attribute graphs (amazon, dblp) - structural-first path
    # dslist = ['amazon', 'dblp']
    
    # # Can specify datasets via command line
    if len(sys.argv) > 1:
        dslist = sys.argv[1:]
    else:
 
        dslist = ['cora','citeseer','pubmed','amazon','dblp','lj1','dblp1']
        # dslist = ['dblp']
        # dslist = ['cora','citeseer']
        # dslist = ['pubmed','amazon','dblp']
        # dslist = ['pubmed','amazon']
        # dslist = ['amazon','dblp']
        # dslist = ['cora','citeseer','pubmed','amazon']
        # dslist = ['dblp']
        # dslist = ['pubmed']
        # dslist = ['cora']
        # dslist = ['dblp']
        
    
    for dataset_name in dslist:
        configds = DATASET_CONFIGS.get(dataset_name)
        if not configds:
            print(f"Dataset {dataset_name} not found in config")
            continue
        
        EDGE_FILE_PATH = configds.get("edge_path")
        COMMUNITY_FILE_PATH = configds.get("community_path")
        network_type = configds["network_type"]
        is_cora = configds.get("is_cora", False)
        
        if is_cora:
            cora_data_dir = configds.get("cora_data_dir")
            cora_dataset = configds.get("cora_dataset")
            if not os.path.exists(cora_data_dir):
                print(f"Warning: Cora data directory not found: {cora_data_dir}")
                continue
        else:
            if not os.path.exists(EDGE_FILE_PATH):
                print(f"Error: Edge file not found: {EDGE_FILE_PATH}")
                continue
            if not os.path.exists(COMMUNITY_FILE_PATH):
                print(f"Warning: Community file not found: {COMMUNITY_FILE_PATH}")
        
        print("="*70)
        print(f"Running HIDC on dataset: {dataset_name}")
        print(f"Type: {'Attribute Graph (HACD)' if is_cora else 'Non-Attribute Graph (Structural Leiden)'}")
        print("="*70)
        
        if is_cora:
            execute_HIDC_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type, 
                                              is_cora=True, cora_data_dir=cora_data_dir, cora_dataset=cora_dataset)
        else:
            execute_HIDC_pipeline_unsupervised(EDGE_FILE_PATH, COMMUNITY_FILE_PATH, network_type, is_cora=False)
