# This is CNN detection on Holidays

from scipy import io
from datetime import datetime, timedelta
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torch.optim as optim

# loads in a dictionary of the matlab variables, as well as a few useless entries
data_path = "../data/24-Oct-2018_data.mat"
loaded_data = io.loadmat(data_path)

# remove useless entries
for e in ['__header__','__globals__','__version__']:
	loaded_data.pop(e)

# convert string datetimes to datetime objects
loaded_data['inc_timestamp'] = [datetime.strptime(d,'%d-%b-%Y %H:%M:%S') for d in loaded_data['inc_timestamp']]
loaded_data['station_times'] = [datetime.strptime(d,'%d-%b-%Y %H:%M:%S') for d in loaded_data['station_times']]

holidays = [
    datetime(2017,1,2),
    datetime(2017,1,16),
    datetime(2017,2,5),
    datetime(2017,2,20),
    datetime(2017,5,29),
    datetime(2017,6,4),
    datetime(2017,9,4),
    datetime(2017,11,23),
    datetime(2017,11,24),
    datetime(2017,12,25),
]

t0 = loaded_data['station_times'][0]
ts = loaded_data['inc_timestamp']
dt = [(t-t0) for t in ts]
t1 = [round(t.days * 288 + t.seconds / 300) for t in dt]
dt = loaded_data['inc_duration'].flatten().tolist()
te = [t + timedelta(minutes=dt[i]) for i,t in enumerate(ts)]
dt = [(t-t0) for t in te]
t2 = [round(t.days * 288 + t.seconds / 300) + 1 for t in dt]

ids = {x[0]:i for i,x in enumerate(list(loaded_data['station_ids'][0]))}
ix = [ids[x[0][0]] for x in loaded_data['inc_station_id']]

data = torch.tensor(np.nan_to_num(loaded_data['station_counts'])).float()
label = torch.zeros(365,1,10, device='cuda')

dmean, dvar = data.mean(dim=0).unsqueeze(0), data.var(dim=0).unsqueeze(0).sqrt()
data.add_(-dmean).div_(dvar * 2)

for h in holidays:
    label[(h-t0).days] = 1


traindata = data[:,:5]
trainlabel = label[:,:,:5]
testdata = data[:,5:]
testlabel = label[:,:,5:]

traindata = traindata.view(365,288,5).permute(0,2,1).contiguous().view(1825,288)#.permute(1,0,2)
traindata = traindata.unsqueeze(1).contiguous().cuda()

trainlabel = trainlabel.view(365,1,5).permute(0,2,1).contiguous().view(1825,1)#.permute(1,0,2)
trainlabel = trainlabel.contiguous().cuda()

testdata = testdata.view(365,288,5).permute(0,2,1).contiguous().view(1825,288)#.permute(1,0,2)
testdata = testdata.unsqueeze(1).contiguous().cuda()
testlabel = testlabel.view(365,1,5).permute(0,2,1).contiguous().view(1825,1)#.permute(1,0,2)
testlabel = testlabel.contiguous().cuda()

def t_acc(net):
    with torch.no_grad():
        out = net(traindata).ge(0.5)
        ans = trainlabel.ge(0.5)
        score = 2 * out.mul(ans).sum().float() / (out.sum() + ans.sum()).float()
        return score.item()

def _acc(net, i=0):
    with torch.no_grad():
        num = float(testlabel.numel())
        out = net(testdata).ge(0.5)
        ans = testlabel.ge(0.5)
        x = out.mul(ans).sum()
        y = out.sum()
        z = ans.sum()
        pcn = x.float().div(y.float()).item()
        #print('\t',x.item(),y.item(),z.item())
        score = 2 * x.float() / (y + z).float()
        print(i,'\t',x.item(),'\t',y.item(),'\t',z.item(),'\t',pcn,'\t', score.item())
        return score.item()

class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv1d( 1,  4, 3, 2)
        self.conv2 = nn.Conv1d( 4,  8, 3, 2)
        self.conv3 = nn.Conv1d( 8, 16, 3, 2)
        self.conv4 = nn.Conv1d(16, 32, 3, 2)
        self.conv5 = nn.Conv1d(32, 64, 3, 2)
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1)
    def forward(self, x):
        x = self.conv1(x).relu()
        x = self.conv2(x).relu()
        x = self.conv4(x).relu()
        x = self.conv5(x).relu().view(x.size(0),-1)
        x = self.fc1(x).relu()
        x = self.fc2(x)
        return x
        
net = CNNClassifier().cuda()

opt = optim.Adam(net.parameters(),lr=1e-3)
sche = optim.lr_scheduler.StepLR(opt, 200, gamma=0.8)

num_of_inc = trainlabel.ge(0.5).sum()

x = traindata
y = trainlabel

for i in range(5000):
    opt.zero_grad()
    out = net(x)
    diff = out - y
    L1 = diff.relu().sum()
    L2 = (-diff).relu().sum()
    L = L1 + L2 * 30
    L.backward()
    opt.step()
    sche.step()
    _acc(net, i

print(_acc(net))
print(t_acc(net))