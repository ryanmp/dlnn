from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SequentialDataSet
from pybrain.structure import SigmoidLayer
from pybrain.structure import LSTMLayer

import itertools
import numpy as np

data = np.loadtxt("datain.txt").T
print data

datain = data[:-1,:] 
dataout = data[1:,:] 

INPUTS = 5
OUTPUTS = 5
HIDDEN = 40

net = buildNetwork(INPUTS, HIDDEN, OUTPUTS, hiddenclass=LSTMLayer, outclass=SigmoidLayer, recurrent=True) 

ds = SequentialDataSet(INPUTS, OUTPUTS)

for x,y in itertools.izip(datain,dataout):
    ds.newSequence()
    ds.appendLinked(tuple(x), tuple(y))

net.randomize()

trainer = BackpropTrainer(net, ds)

for _ in range(1000):
    print trainer.train()