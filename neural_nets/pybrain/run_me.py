'''
Author: Ryan Phillips
Created: 9-14-2014
Last Modified 9-14-2014

'''

# global libraries
import scipy, numpy, random, math, logging, seaborn, matplotlib.pyplot, datetime


# pybrain
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,UnsupervisedDataSet
from pybrain.structure import LinearLayer
from pybrain.optimization.populationbased.ga import GA
from pybrain.structure.modules import TanhLayer

# global variables
num_inputs = 70
num_hidden = 800
num_outputs = 1

columns = 71 #total

training_time = 100

in_sample_len = 90
out_sample_len = 20



def main():
    logging.basicConfig(filename='run_me.log', level=logging.DEBUG, format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',)

    data_all, d = process_data()

    net = init_ann(num_inputs,num_hidden,num_outputs)

    #training
    print 'training on: 0 -', in_sample_len

    '''
    ds = SupervisedDataSet(num_inputs, num_outputs)

    for i in xrange(in_sample_len):
        x = d[(1,1)][i]
        sample_in = x[1:-1]
        sample_target = x[:1]
        ds.addSample(sample_in,sample_target)

    ga = GA(ds.evaluateModuleMSE, net, minimize=True)
    for i in range(1000):
        net = ga.learn(0)[0]
    '''

    # training
    ds = SupervisedDataSet(num_inputs, num_outputs)
    for i in xrange(in_sample_len):
        x = d[(1,1)][i]
        sample_in = x[1:-1]
        sample_target = x[:1]
        ds.addSample(sample_in,sample_target)

    train_ann(net,ds)


    #using trained model
    out = []
    print 'running on:', in_sample_len,out_sample_len+in_sample_len
    for i in xrange(in_sample_len,out_sample_len+in_sample_len):
        x = d[(1,1)][i]
        sample_in = x[1:-1]
        out_row = []
        out_row.append(use_trained_ann(net,sample_in)[0])
        out_row.append(x[-1])
        out.append( out_row )
    out = numpy.array(out)

    graph_it2(d,out)


    return data_all, d, out, ds

def process_data():
    datafile = open('../../contest_data_weekly.csv')

    data = []
    d = {} #init dict data structure with our data

    # pairs included in this dataset (retailer, item)
    keys = [(1, 2), (1, 3), (3, 3), (2, 1), (2, 3), (4, 3), (2, 2), (4, 2), (1, 1)]
    for i in keys:
        d[i] = []

    for idx, row in enumerate(datafile):

        if idx%100 == 0:
            print idx

        data_strings = row.strip().split(',')
        data_floats = [0.0 for i in xrange(columns)]

        for i in xrange(columns):
            # next I was planning on converting them all to floats,
            # but what should I do with the NULLs?
            try:
                tmp_float = float(data_strings[i])
            except:
                tmp_float = 0.0 # probably a bad idea to just use zero...

            data_floats[i] = tmp_float

        dt = iso_to_gregorian(int(data_floats[3]), int(data_floats[4]), 0)

        data_floats.append(dt)

        # excluding the out of sample data
        if (data_floats[0] < 5000):
            data.append(data_floats)
            # also add to our dict
            d[(data_floats[1],data_floats[2])].append(data_floats)


    # convert all arrays to numpy arrays
    data = numpy.array(data)
    for i in d.keys():
        d[i] = numpy.array(d[i])

    return data, d




def init_ann(num_inputs,num_hidden,num_outputs):
    net = buildNetwork(num_inputs, num_hidden, num_outputs, outclass=LinearLayer,bias=True, hiddenclass=TanhLayer)
    return net

def train_ann(net,ds):

    trainer = BackpropTrainer( net, dataset=ds, verbose=True, learningrate = 0.0005)
    trainer.trainEpochs(20)


def use_trained_ann(net,ds):
    ts = UnsupervisedDataSet(num_inputs,)
    ts.addSample(ds)
    x = [ i for i in net.activateOnDataset(ts)[0]]
    return x

def graph_it(d):
    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(111)
    ax1.xaxis_date()

    for i in d.keys():
        ax1.plot((d[i][:,-1]), d[i][:,0], 'o-',  ms=1.5, linewidth=.6)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()


def graph_it2(d,out):
    fig = matplotlib.pyplot.figure()
    ax1 = fig.add_subplot(111)
    ax1.xaxis_date()

    out = numpy.array(out)

    i = (1,1)
    ax1.plot(
        (d[i][:in_sample_len+out_sample_len,-1]),
        d[i][:in_sample_len+out_sample_len,0],
        'o-',  ms=1.5, linewidth=.6
            )

    ax1.plot(out[:,1],out[:,0], 'o-',  ms=1.5, linewidth=.6)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.show()





def iso_year_start(iso_year):
    "The gregorian calendar date of the first day of the given ISO year"
    fourth_jan = datetime.date(iso_year, 1, 4)
    delta = datetime.timedelta(fourth_jan.isoweekday()-1)
    return fourth_jan - delta

def iso_to_gregorian(iso_year, iso_week, iso_day):
    "Gregorian calendar date for the given ISO year, week and day"
    year_start = iso_year_start(iso_year)
    return year_start + datetime.timedelta(days=iso_day-1, weeks=iso_week-1)





if __name__ == '__main__': data_all, d, out, ds = main()