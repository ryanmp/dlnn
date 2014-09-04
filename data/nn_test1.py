import numpy # better arrays
import cPickle as pickle # persistant storage type

print 'loading data from pickle'
data = pickle.load( open( "data.p", "rb" ) )