import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder
from torch.utils.data.distributed import DistributedSampler
import re
import random
import nltk
from nltk.tokenize import word_tokenize
from config import Config

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def clean_text(text):
    """文本清理"""
    # 转换为小写
    text = text.lower()
    # 移除HTML标签
    text = re.sub(r'<[^>]+>', '', text)
    # 移除URL
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # 移除特殊字符和多余空格
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def text_augmentation(text):
    """文本增强方法"""
    # 分离标题和描述
    parts = text.split('[SEP]')
    if len(parts) != 2:
        return text
        
    headline = parts[0].strip()
    description = parts[1].strip()
    
    # 分别对标题和描述进行增强
    headline_tokens = word_tokenize(headline)
    desc_tokens = word_tokenize(description)
    
    # 随机选择一种增强方法
    aug_type = random.choice(['swap', 'delete', 'back_translate'])
    
    if aug_type == 'swap' and len(desc_tokens) > 3:
        # 只在描述中进行词交换，保护标题
        idx = random.randint(0, len(desc_tokens)-2)
        # 确保不交换标点符号
        if desc_tokens[idx].isalnum() and desc_tokens[idx+1].isalnum():
            desc_tokens[idx], desc_tokens[idx+1] = desc_tokens[idx+1], desc_tokens[idx]
    
    elif aug_type == 'delete' and len(desc_tokens) > 5:
        # 只在描述中删除，且避免删除过多
        num_to_delete = min(2, len(desc_tokens) // 10)  # 最多删除10%的词
        for _ in range(num_to_delete):
            idx = random.randint(0, len(desc_tokens)-1)
            # 确保不删除重要词（如：名词、动词的开头）
            if not desc_tokens[idx][0].isupper() and desc_tokens[idx].isalnum():
                desc_tokens.pop(idx)
    
    elif aug_type == 'back_translate':
        # 模拟回译效果：随机替换一些常见词
        common_replacements = {
            'said': 'stated',
            'big': 'large',
            'small': 'tiny',
            'good': 'great',
            'bad': 'poor',
            'important': 'significant',
            'problem': 'issue',
            'change': 'modify',
            'help': 'assist',
            'show': 'display'
        }
        
        # 只在描述中进行替换
        for i, token in enumerate(desc_tokens):
            if token.lower() in common_replacements and random.random() < 0.5:
                desc_tokens[i] = common_replacements[token.lower()]
    
    # 重新组合文本
    augmented_headline = ' '.join(headline_tokens)
    augmented_description = ' '.join(desc_tokens)
    
    return f"{augmented_headline} [SEP] {augmented_description}"

class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, is_training=True):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_training = is_training
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # 清理文本
        text = clean_text(text)
        
        # 训练时进行数据增强，降低概率到20%
        if self.is_training and random.random() < 0.2:
            text = text_augmentation(text)
        
        # 使用动态填充
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def create_data_loaders(world_size=None, rank=None):
    # 加载数据
    with open(Config.DATA_FILE, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    
    # 提取文本和标签
    texts = []
    categories = []
    
    for item in data:
        # 合并标题和描述，添加分隔符
        text = f"{item['headline']} [SEP] {item['short_description']}"
        texts.append(text)
        categories.append(item['category'])
    
    # 编码标签
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(categories)
    
    # 更新配置中的类别数
    Config.NUM_CLASSES = len(label_encoder.classes_)
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME, local_files_only=False)
    
    # 计算训练集和验证集的大小
    total_size = len(texts)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    # 分割数据集
    indices = list(range(total_size))
    random.seed(Config.SEED)
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # 创建训练集和验证集
    train_texts = [texts[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_texts = [texts[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]
    
    # 创建数据集
    train_dataset = NewsDataset(train_texts, train_labels, tokenizer, Config.MAX_LENGTH, is_training=True)
    val_dataset = NewsDataset(val_texts, val_labels, tokenizer, Config.MAX_LENGTH, is_training=False)
    
    # 创建数据加载器，支持分布式训练
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False
        )
    else:
        train_sampler = None
        val_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=(train_sampler is None),
        num_workers=Config.NUM_WORKERS,
        sampler=train_sampler,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        sampler=val_sampler,
        pin_memory=True
    )
    
    return train_loader, val_loader, label_encoder