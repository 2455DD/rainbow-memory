import torch.nn as nn
from models.layers import FinalBlock

class RNNNet(nn.Module):
    def __init__(self, opt) -> None:
        super(RNNNet,self).__init__()
        self.initial = nn.Conv1d(4,16,5)# CNN临时大小
        self.rnn = nn.RNN(4,16,5,batch_first=True)       # RNN临时层数，Watchout
        self.fc = FinalBlock(opt=opt,in_channels=32)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self,x):
        out = self.initial(x)   #B*16*Lout
        out = out.permute(0,2,1)#B*16*Lout -> B*Lout*16
        out = self.rnn(x) #B*Lout*16 -> B*Lout*32
        out = self.pool(out)
        out = self.fc(out)