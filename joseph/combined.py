import torch
import torch.nn as nn
from multiple import MetricLSTM  

# class CombinedModel(nn.Module): #this one is old
#     def __init__(self):
#         super().__init__()
#         self.bal = MetricLSTM()
#         self.speed = MetricLSTM()
#         self.dev = MetricLSTM()

#     def load_weights(self, imb_path, spd_path, dev_path, device):
#         self.bal.load_state_dict(torch.load(imb_path, map_location=device))
#         self.speed.load_state_dict(torch.load(spd_path, map_location=device))
#         self.dev.load_state_dict(torch.load(dev_path, map_location=device))
        
#         #keep in eval and freeze weights and want to keep fully trained and not interfere with each other
#         for param in self.parameters():
#             param.requires_grad = False
#         self.eval()

#     def forward(self, x, lengths):
#         out_imb = self.bal(x, lengths)
#         out_spd = self.speed(x, lengths)
#         out_dev = self.dev(x, lengths)

#         return out_imb, out_spd, out_dev
    
    
class FGA_Estimator(nn.Module): #the new one with all of the bath metrics 
    def __init__(self, imb_p, spd_p, lat_p, gait_dev_p, device_p):
        super().__init__()
 
        self.imb = MetricLSTM()
        self.spd = MetricLSTM()
        self.lat = MetricLSTM()
        self.gait_dev = MetricLSTM()
        self.device = MetricLSTM()

        self.imb.load_state_dict(torch.load(imb_p))
        self.spd.load_state_dict(torch.load(spd_p))
        self.lat.load_state_dict(torch.load(lat_p))
        self.gait_dev.load_state_dict(torch.load(gait_dev_p))
        self.device.load_state_dict(torch.load(device_p))
 
        for param in self.parameters(): #freeze weights
            param.requires_grad = False

        self.fga_head = nn.Linear(5, 1)

    def forward(self, x, lengths):
        with torch.no_grad():
            s1 = self.imb(x, lengths)
            s2 = self.spd(x, lengths)
            s3 = self.lat(x, lengths)
            s4 = self.gait_dev(x, lengths)
            s5 = self.device(x, lengths)

        combined_features = torch.stack([s1, s2, s3, s4, s5], dim=1)
        
        return self.fga_head(combined_features)