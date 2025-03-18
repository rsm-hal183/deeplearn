import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import pandas as pd

from config import Config
from dataset import create_data_loaders
from model import NewsClassifier
from pprint import pprint

def train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch):
    model.train()
    total_loss = 0
    
    if isinstance(train_loader.sampler, DistributedSampler):
        train_loader.sampler.set_epoch(epoch)
    
    progress_bar = tqdm(train_loader, desc='Training', disable=not dist.get_rank() == 0)
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        if dist.get_rank() == 0:
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Evaluating', disable=not dist.get_rank() == 0):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions.extend(outputs.argmax(dim=1).cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # 在分布式环境中收集所有进程的预测结果
    if dist.is_initialized():
        all_predictions = [None for _ in range(dist.get_world_size())]
        all_labels = [None for _ in range(dist.get_world_size())]
        
        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_labels, true_labels)
        
        if dist.get_rank() == 0:
            predictions = []
            true_labels = []
            for pred, label in zip(all_predictions, all_labels):
                predictions.extend(pred)
                true_labels.extend(label)
    
    val_loss = total_loss / len(val_loader)
    
    if dist.get_rank() == 0:
        # 计算详细的评估指标
        accuracy = accuracy_score(true_labels, predictions)
        
        # 计算每个类别的precision, recall, f1
        precision, recall, f1, support = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            zero_division=0,  # 处理没有预测样本的类别
            average=None
        )
        
        # 计算总体指标
        macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            zero_division=0,
            average='macro'
        )
        
        weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
            true_labels, 
            predictions, 
            zero_division=0,
            average='weighted'
        )
        
        # 生成分类报告
        class_report = classification_report(
            true_labels, 
            predictions, 
            digits=4,
            zero_division=0
        )
        
        # 创建评估指标字典
        metrics = {
            'val_loss': val_loss,
            'accuracy': accuracy,
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1,
            'weighted_precision': weighted_precision,
            'weighted_recall': weighted_recall,
            'weighted_f1': weighted_f1,
            'classification_report': class_report
        }
        
        # 打印评估结果
        print(f"\nValidation Loss: {val_loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nMacro Metrics:")
        print(f"Precision: {macro_precision:.4f}")
        print(f"Recall: {macro_recall:.4f}")
        print(f"F1: {macro_f1:.4f}")
        print(f"\nWeighted Metrics:")
        print(f"Precision: {weighted_precision:.4f}")
        print(f"Recall: {weighted_recall:.4f}")
        print(f"F1: {weighted_f1:.4f}")
        print("\nClassification Report:")
        print(class_report)
        
        return val_loss, metrics
    
    return val_loss, None

def setup_distributed():
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        
        return local_rank, world_size
    return 0, 1

def main():
    # 设置分布式训练
    local_rank, world_size = setup_distributed()
    
    # 设置设备
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    Config.DEVICE = device
    
    # 设置随机种子
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)
    
    # 创建数据加载器（使用DistributedSampler）
    train_loader, val_loader, label_encoder = create_data_loaders(world_size=world_size, rank=local_rank)
    
    # 打印数据集信息
    if local_rank == 0:  # 只在主进程打印信息
        print(vars(Config))
        
        print("\n" + "="*50)
        print("数据集信息统计")
        print("="*50)
        
        # 计算数据集大小
        train_size = len(train_loader.dataset)
        val_size = len(val_loader.dataset)
        total_size = train_size + val_size
        
        print(f"\n总样本数: {total_size}")
        print(f"训练集样本数: {train_size}")
        print(f"验证集样本数: {val_size}")
        print(f"批次大小: {Config.BATCH_SIZE}")
        print(f"训练批次数: {len(train_loader)}")
        print(f"验证批次数: {len(val_loader)}")
        
        # 打印类别信息
        print(f"\n类别数量: {len(label_encoder.classes_)}")
        print("\n类别列表:")
        for i, label in enumerate(label_encoder.classes_):
            print(f"{i}: {label}")
            
        print("\n" + "="*50 + "\n")
    
    # 创建模型
    model = NewsClassifier(Config.NUM_CLASSES).to(device)
    
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank])
    
    # 创建优化器和学习率调度器
    optimizer = AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        eps=Config.ADAM_EPSILON,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 计算总训练步数和warmup步数
    total_steps = len(train_loader) * Config.EPOCHS
    warmup_steps = int(total_steps * Config.WARMUP_RATIO)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # 创建损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 只在主进程创建保存目录
    if local_rank == 0:
        os.makedirs(os.path.dirname(Config.MODEL_SAVE_PATH), exist_ok=True)
    
    # 训练循环
    best_val_loss = float('inf')
    for epoch in range(Config.EPOCHS):
        if local_rank == 0:
            print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, criterion, device, epoch)
        if local_rank == 0:
            print(f"Average training loss: {train_loss:.4f}")
        
        # 评估
        val_loss, metrics = evaluate(model, val_loader, criterion, device)
        if local_rank == 0:
            print(f"Validation loss: {val_loss:.4f}")
            print("\nMetrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value}")
        
        # 只在主进程保存模型
        if local_rank == 0 and val_loss < best_val_loss:
            best_val_loss = val_loss
            model_to_save = model.module if hasattr(model, 'module') else model
            
            # 保存更多训练信息
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model_to_save.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'metrics': metrics,
                'label_encoder': label_encoder,
                'config': {k: v for k, v in vars(Config).items() if not k.startswith('__')}  # 将Config转换为普通字典
            }
            
            # 保存检查点
            torch.save(save_dict, Config.MODEL_SAVE_PATH)
            print(f"Saved best model to {Config.MODEL_SAVE_PATH}")
            
            # 保存评估指标
            if metrics:
                metrics_file = os.path.join(os.path.dirname(Config.MODEL_SAVE_PATH), 'best_metrics.txt')
                with open(metrics_file, 'w') as f:
                    for metric, value in metrics.items():
                        if isinstance(value, (float, int)):
                            f.write(f"{metric}: {value}\n")
                        elif metric == 'classification_report':
                            f.write(f"\nClassification Report:\n{value}\n")
    
    # 清理分布式进程组
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    # pprint(vars(Config))
    main() 