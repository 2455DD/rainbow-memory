import torch.nn as nn
import logging
import torch
logger = logging.getLogger()

class FinalBlock2(nn.Module):
    def __init__(self, opt,in_channels,out_channels) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_features = in_channels
        self.out_features = out_channels
        self.fc = nn.Conv1d(in_channels = opt.width,out_channels= opt.num_classes,kernel_size=1,bias=False)
        self.bn = nn.BatchNorm1d(opt.num_classes,affine=opt.affine_bn, eps=opt.bn_eps)
        self.pool = nn.AdaptiveAvgPool1d(1)
    def forward(self,x):
        out =x 
        logger.debug(f"before fc x.shape:{out.shape}(opt.width ={self.width} )")
        out = self.fc(out)
        logger.debug(f"after fc x.shape:{out.shape}(num_classes:{self.num_classes})")
        out = out.permute(0,2,1) #B*Lout*num_classes -> B*num_classes*Lout
        logger.debug(f"before bn x.shape:{out.shape}(num_classes:{self.num_classes})")
        out = self.bn(out) 
        return out

class RNNNet(nn.Module):
    def __init__(self, opt) -> None:
        super(RNNNet,self).__init__()
        
        self.num_classes = opt.num_classes
        logger.debug(f"opt.num_classes:{opt.num_classes}")

        self.width = opt.width
        self.initial = nn.Conv1d(4,opt.width,1)# CNN临时大小,opt.width=16
        self.rnn = nn.RNN(opt.width,opt.width,5,batch_first=True)       # RNN临时层数，Watchout
        # self.fc = nn.Linear(in_features = opt.width,out_features =opt.num_classes,bias=False)
        self.fc = FinalBlock2(opt,opt.width,opt.num_classes)
        self.act = nn.ReLU()
        
        
    def forward(self,x):
        logger.debug(f"x.shape:{x.shape}")
        # out = self.initial(x.permute(0,2,1))   #B*4*Lin   -> B*Lout*4 -> B*Lout*4
        out = self.initial(x)   #B*4*Lin   -> B*Lout*4 -> B*Lout*4
        out = out.permute(0,2,1)#B*16*Lout -> B*Lout*16
        logger.debug(f"before rnn x.shape:{out.shape}")
        out,_ = self.rnn(out) #B*Lout*16 -> B*Lout*16
        # out = self.pool(out)
        out = out.permute(0,2,1) #B*Lout*num_classes -> B*num_classes*Lout
        logger.debug(f"before fc x.shape:{out.shape}")
        out = self.fc(out)
        fin_out = self.act(out)
        logging.debug(f"out shape:{fin_out.shape}")
        return torch.softmax(fin_out,dim=1)