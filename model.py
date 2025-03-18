import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from config import Config

class NewsClassifier_base(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.bert = AutoModel.from_pretrained(Config.MODEL_NAME)
        self.dropout = nn.Dropout(Config.DROPOUT)
        self.classifier = nn.Linear(Config.HIDDEN_SIZE, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits 

class NewsClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # BERT层
        self.bert = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR,
            local_files_only=False,  # 允许从网络下载
            output_hidden_states=True  # 输出所有层的隐藏状态
        )
        
        # 特征提取层
        self.conv1 = nn.Conv1d(Config.HIDDEN_SIZE, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(Config.HIDDEN_SIZE, 256, kernel_size=5, padding=2)
        
        # 多头注意力层
        self.attention = nn.MultiheadAttention(Config.HIDDEN_SIZE, num_heads=8, dropout=Config.DROPOUT)
        
        # 特征融合层
        hidden_size = Config.HIDDEN_SIZE
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 4 + 512, hidden_size),  # 4 * hidden_size + 2 * 256(conv特征)
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # 分类层
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 1. BERT输出
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output        # [batch_size, hidden_size]
        hidden_states = outputs.hidden_states        # tuple of [batch_size, seq_len, hidden_size]
        
        # 2. 获取最后四层的隐藏状态并加权平均
        last_4_hidden = torch.stack(hidden_states[-4:])  # [4, batch_size, seq_len, hidden_size]
        weighted_hidden = torch.mean(last_4_hidden, dim=0)  # [batch_size, seq_len, hidden_size]
        
        # 3. 卷积特征提取
        # 转换维度以适应卷积操作 [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
        conv_input = weighted_hidden.transpose(1, 2)
        # 使用不同大小的卷积核提取特征
        conv1_output = F.relu(self.conv1(conv_input))  # [batch_size, 256, seq_len]
        conv2_output = F.relu(self.conv2(conv_input))  # [batch_size, 256, seq_len]
        # 最大池化提取最显著的特征
        conv1_pooled = F.max_pool1d(conv1_output, conv1_output.size(2)).squeeze(2)  # [batch_size, 256]
        conv2_pooled = F.max_pool1d(conv2_output, conv2_output.size(2)).squeeze(2)  # [batch_size, 256]
        # 合并卷积特征
        conv_features = torch.cat([conv1_pooled, conv2_pooled], dim=1)  # [batch_size, 512]
        
        # 4. 自注意力处理
        attended_output, _ = self.attention(
            weighted_hidden.transpose(0, 1),
            weighted_hidden.transpose(0, 1),
            weighted_hidden.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attended_output = attended_output.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # 5. 池化操作
        # 最大池化
        max_pooled = torch.max(attended_output * attention_mask.unsqueeze(-1), dim=1)[0]
        # 平均池化
        avg_pooled = torch.sum(attended_output * attention_mask.unsqueeze(-1), dim=1) / \
                    torch.sum(attention_mask, dim=1, keepdim=True)
        
        # 6. 特征融合
        # 拼接所有特征
        concat_features = torch.cat([
            pooled_output,          # BERT的[CLS]表示
            max_pooled,             # 最大池化特征
            avg_pooled,             # 平均池化特征
            torch.max(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0],  # 原始最后层的最大池化
            conv_features           # 卷积特征
        ], dim=1)
        
        # 融合特征
        fused_features = self.feature_fusion(concat_features)
        
        # 7. 分类
        logits = self.classifier(fused_features)
        
        return logits 