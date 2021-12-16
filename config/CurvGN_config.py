class Config():
    def __init__(self):
        self.is_DiGraph = False                # 是否是有向图
        self.times = 1                         # 实验次数
        self.epoch_num = 10000                  # 每次实验进行的轮数
        self.patience = 1000                    # early stop所需轮数
        self.d_names = "CUB"                   # 实验所用数据集
        self.learning_rate = 0.01             # 学习率
        self.gamma1 = 2e-6                     # Reg1权重
        self.gamma2 = 3e-7                     # Reg2权重
        self.leaky_relu_negative_slope = 0.1   # leaky relu 负斜率
        self.loss_mode = 0                     # 0:cross_entropy 1:cross_entropy + gamma1*Reg1 + gamma2*Reg2
        self.curvature_activate_mode_Ollivier = 3       # 0:全连接层 1:ReLU 2:Leaky ReLU 3:PReLU single value 4:PReLU all channel 5:ELU 6:Tanh 7:Sigmoid
        self.curvature_activate_mode_Forman = 2
        self.curvature_activate_mode = 0
        self.mask_mode = 1                     # 0:随机mask 1:ricci curvature 从低到高mask 2:ricci curvature 从高到低mask
        self.mask_rate = 0.0                   # mask掉的比例
        self.learnable_curvature = 0           # 是否采用可学习的参数来加权node message  0:使用curvature加权 1:采用随机数加权 2:采用可学习参数加权 3:采用边上两个节点的余弦相似度加权
        self.curvature_method = 2              # 0:Ollivier 1:Forman 2:Ollivier+Forman
        self.method = "ConvCurv"
