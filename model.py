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
        # BERT��
        self.bert = AutoModel.from_pretrained(
            Config.MODEL_NAME,
            cache_dir=Config.MODEL_CACHE_DIR,
            local_files_only=False,  # �������������
            output_hidden_states=True  # ������в������״̬
        )
        
        # ������ȡ��
        self.conv1 = nn.Conv1d(Config.HIDDEN_SIZE, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(Config.HIDDEN_SIZE, 256, kernel_size=5, padding=2)
        
        # ��ͷע������
        self.attention = nn.MultiheadAttention(Config.HIDDEN_SIZE, num_heads=8, dropout=Config.DROPOUT)
        
        # �����ںϲ�
        hidden_size = Config.HIDDEN_SIZE
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_size * 4 + 512, hidden_size),  # 4 * hidden_size + 2 * 256(conv����)
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT)
        )
        
        # �����
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        # 1. BERT���
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        pooled_output = outputs.pooler_output        # [batch_size, hidden_size]
        hidden_states = outputs.hidden_states        # tuple of [batch_size, seq_len, hidden_size]
        
        # 2. ��ȡ����Ĳ������״̬����Ȩƽ��
        last_4_hidden = torch.stack(hidden_states[-4:])  # [4, batch_size, seq_len, hidden_size]
        weighted_hidden = torch.mean(last_4_hidden, dim=0)  # [batch_size, seq_len, hidden_size]
        
        # 3. ���������ȡ
        # ת��ά������Ӧ������� [batch_size, seq_len, hidden_size] -> [batch_size, hidden_size, seq_len]
        conv_input = weighted_hidden.transpose(1, 2)
        # ʹ�ò�ͬ��С�ľ������ȡ����
        conv1_output = F.relu(self.conv1(conv_input))  # [batch_size, 256, seq_len]
        conv2_output = F.relu(self.conv2(conv_input))  # [batch_size, 256, seq_len]
        # ���ػ���ȡ������������
        conv1_pooled = F.max_pool1d(conv1_output, conv1_output.size(2)).squeeze(2)  # [batch_size, 256]
        conv2_pooled = F.max_pool1d(conv2_output, conv2_output.size(2)).squeeze(2)  # [batch_size, 256]
        # �ϲ��������
        conv_features = torch.cat([conv1_pooled, conv2_pooled], dim=1)  # [batch_size, 512]
        
        # 4. ��ע��������
        attended_output, _ = self.attention(
            weighted_hidden.transpose(0, 1),
            weighted_hidden.transpose(0, 1),
            weighted_hidden.transpose(0, 1),
            key_padding_mask=~attention_mask.bool()
        )
        attended_output = attended_output.transpose(0, 1)  # [batch_size, seq_len, hidden_size]
        
        # 5. �ػ�����
        # ���ػ�
        max_pooled = torch.max(attended_output * attention_mask.unsqueeze(-1), dim=1)[0]
        # ƽ���ػ�
        avg_pooled = torch.sum(attended_output * attention_mask.unsqueeze(-1), dim=1) / \
                    torch.sum(attention_mask, dim=1, keepdim=True)
        
        # 6. �����ں�
        # ƴ����������
        concat_features = torch.cat([
            pooled_output,          # BERT��[CLS]��ʾ
            max_pooled,             # ���ػ�����
            avg_pooled,             # ƽ���ػ�����
            torch.max(last_hidden_state * attention_mask.unsqueeze(-1), dim=1)[0],  # ԭʼ��������ػ�
            conv_features           # �������
        ], dim=1)
        
        # �ں�����
        fused_features = self.feature_fusion(concat_features)
        
        # 7. ����
        logits = self.classifier(fused_features)
        
        return logits 