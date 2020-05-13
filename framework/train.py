# -*- coding: utf-8 -*-
#import ngraph_bridge
import tensorflow.compat.v1 as tf
from agent.server import train

def main(argv):
	train()
	
if __name__ == '__main__':
	print('TensorFlow version: ',tf.__version__)
    #print(ngraph_bridge.__version__)
	tf.app.run()