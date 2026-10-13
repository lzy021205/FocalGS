import torch
import torch.nn as nn

class GTnet(nn.Module):
    #把默认的输入维度从 time_dim(32) 改成 38 (32 + 3维位置 + 3维朝向)
    def __init__(self, in_dim=38, width=64):
        """
        Physics-based Camera Parameter Network (物理相机网络)
        """
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.ReLU(),
            nn.Linear(width, width),
            nn.ReLU(),
            nn.Linear(width, 2) # 依然死死锁在 2 个物理输出：[光圈, 对焦面]
        ).to("cuda")

        # 优雅的初始化策略不变
        nn.init.xavier_normal_(self.mlp[-1].weight, 0.01)
        nn.init.constant_(self.mlp[-1].bias, 0.0)
            
    def forward(self, camera_state):
        """
        Args:
            camera_state: [1, 38] 包含了 时间特征 + 相机坐标 + 相机朝向
        """
        phys_params = self.mlp(camera_state)
        return phys_params