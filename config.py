class Config1:
    # 数据相关配置
    DATA_FILE = "News_Category_Dataset_v3.json"
    MAX_LENGTH = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # 模型相关配置
    MODEL_NAME = "bert-base-uncased"
    MODEL_CACHE_DIR = "./model_cache"  # 指定本地缓存目录
    HIDDEN_SIZE = 768
    NUM_CLASSES = None  # 将在运行时根据数据集确定
    DROPOUT = 0.2  # 增加dropout
    
    # 训练相关配置
    LEARNING_RATE = 2e-5
    WARMUP_RATIO = 0.1  # 添加warmup
    EPOCHS = 5
    GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积
    WEIGHT_DECAY = 0.01
    DEVICE = "cuda"  # 如果没有GPU，会自动切换到CPU
    
    # 优化器配置
    ADAM_EPSILON = 1e-8
    MAX_GRAD_NORM = 1.0
    
    # 学习率调度器配置
    SCHEDULER_TYPE = 'linear'  # 可选: linear, cosine
    NUM_CYCLES = 0.5  # 用于cosine调度器
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 2
    
    # 路径配置
    MODEL_SAVE_PATH = "models/news_classifier.pt"
    
    # 随机种子
    SEED = 42 
class Config:
    # 数据相关配置
    DATA_FILE = "News_Category_Dataset_v3.json"
    MAX_LENGTH = 512  # 增加序列长度，捕获更多文本信息
    BATCH_SIZE = 16   # 减小批次大小，提高模型稳定性
    NUM_WORKERS = 4
    
    # 模型相关配置
    MODEL_NAME = "bert-base-uncased"  # 也可以考虑使用 "roberta-base"
    MODEL_CACHE_DIR = "./model_cache"
    HIDDEN_SIZE = 768
    NUM_CLASSES = None
    DROPOUT = 0.3     # 增加dropout防止过拟合
    
    # 训练相关配置
    LEARNING_RATE = 1e-5  # 降低学习率
    WARMUP_RATIO = 0.2    # 增加warmup比例
    EPOCHS = 5           # 增加训练轮数
    GRADIENT_ACCUMULATION_STEPS = 4  # 增加梯度累积步数
    WEIGHT_DECAY = 0.02   # 增加权重衰减
    DEVICE = "cuda"
    
    # 优化器配置
    ADAM_EPSILON = 1e-8
    MAX_GRAD_NORM = 1.0
    
    # 学习率调度器配置
    SCHEDULER_TYPE = 'cosine'  # 使用cosine调度器
    NUM_CYCLES = 0.5
    
    # 早停配置
    EARLY_STOPPING_PATIENCE = 3  # 增加早停耐心值
    
    # 路径配置
    MODEL_SAVE_PATH = "models/news_classifier.pt"
    
    # 随机种子
    SEED = 42 