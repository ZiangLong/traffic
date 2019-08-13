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

# t0 = starting time = 2017.1.1 00:00
t0 = loaded_data['station_times'][0]
# Time_Start: The time when incident happens
ts = loaded_data['inc_timestamp']
# type(dt) = [timedelta], dt = time duration between t0 and ts
dt = [(t-t0) for t in ts]
# type(t1) = [int], t1 = index of Time_Start
t1 = [round(t.days * 288 + t.seconds / 300) for t in dt]
# type(dt) = [int], incident duration in minutes
dt = loaded_data['inc_duration'].flatten().tolist()
# Time_End: The time when Incident End
te = [t + timedelta(minutes=dt[i]) for i,t in enumerate(ts)]
# type(dt) = [timedelta], dt = time duration between t0 and te
dt = [(t-t0) for t in te]
# type(t2) = [int]. i2 = index of Time_End
t2 = [round(t.days * 288 + t.seconds / 300) + 1 for t in dt]

ids = {x[0]:i for i,x in enumerate(list(loaded_data['station_ids'][0]))}
ix = [ids[x[0][0]] for x in loaded_data['inc_station_id']]

# Set missing entries to zero
device = 'cuda' if torch.cuda.is_available() else 'cpu'

data = torch.tensor(np.nan_to_num(loaded_data['station_counts'])).float()
label = torch.zeros(data.size(), device=device)
for i, x in enumerate(ix):
    label[t1[i]:t2[i],x]=1

# Normalized traffic data, set mean = 0, std var = 1
dmean, dvar = data.mean(dim=0).unsqueeze(0), data.var(dim=0).unsqueeze(0).sqrt()
data.add_(-dmean).div_(dvar * 2)

# Separate Data, half training half testing
traindata = data[:,:5]
trainlabel = label[:,:5]
testdata = data[:,5:]
testlabel = label[:,5:]

# Dimension = (288 records a day) *(365 days * 5 sensors) * 1
traindata = traindata.view(365,288,5).permute(0,2,1).contiguous().view(1825,288,1).permute(1,0,2)
traindata = traindata.contiguous()

# Dimension = (288 records a day) *(365 days * 5 sensors) * 1
trainlabel = trainlabel.view(365,288,5).permute(0,2,1).contiguous().view(1825,288,1).permute(1,0,2)
trainlabel = trainlabel.contiguous()

testdata = testdata.view(365,288,5).permute(0,2,1).contiguous().view(1825,288,1).permute(1,0,2)
testdata = testdata.contiguous()
testlabel = testlabel.view(365,288,5).permute(0,2,1).contiguous().view(1825,288,1).permute(1,0,2)
testlabel = testlabel.contiguous()

if device == 'cuda':
    traindata = traindata.cuda()
    trainlabel = trainlabel.cuda()
    testdata = testdata.cuda()
    testlabel = testlabel.cuda()

# Number of Layers
nol = 2

class BRNNClassifier(nn.Module):
    def __init__(self, hidden_dim, output_size):
        super(BRNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(1, hidden_dim, nol, bidirectional=True)
        self.hidden2out = nn.Linear(hidden_dim * 2, output_size)
        self.softmax = nn.Softmax()
    def forward(self, x):
        batch_size = x.size(1)
        h = torch.zeros(nol * 2, batch_size, self.hidden_dim, device=device)
        self.hidden = h
        out, hn = self.rnn(x, self.hidden)
        out = self.hidden2out(out.relu())
        return out.sigmoid()

# Training Accuracy
def t_acc(net):
    with torch.no_grad():
        out = net(traindata).ge(0.5)
        ans = trainlabel.ge(0.5)
        score = 2 * out.mul(ans).sum().float() / (out.sum() + ans.sum()).float()
        return score.item()

# Testing Accuracy
def _acc(net, i=0):
    with torch.no_grad():
        num = float(testlabel.numel())
        out = net(testdata).ge(0.5)
        ans = testlabel.ge(0.5)
        x = out.mul(ans).sum()
        y = out.sum()
        z = ans.sum()
        pcn = x.float().div(y.float()).item()
        score = 2 * x.float() / (y + z).float()
        print(i,'\t',x.item(),'\t',y.item(),'\t',z.item(),'\t',pcn,'\t', score.item())
        return score.item()

net = BRNNClassifier(128,1)

if device == 'cuda':
    net = net.cuda()

#opt = optim.SGD(net.parameters(),lr=1e-6,momentum=0.9)
opt = optim.Adam(net.parameters(),lr=1e-5)
sche = optim.lr_scheduler.StepLR(opt, 200, gamma=0.95)

num_of_inc = trainlabel.ge(0.5).sum()

# for notation short
x = traindata
y = trainlabel


for i in range(2000):
    opt.zero_grad()
    out = net(x)
    diff = y - out
    L1 = diff.relu().sum()
    L2 = (-diff).relu().sum()
    L = L1 * 2500 + L2
    L.backward()
    opt.step()
    sche.step()
    _acc(net, i)
