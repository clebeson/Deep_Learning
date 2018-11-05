from base.baselayer import BaseLayer
import tensorflow as tf



class EnsambleMean(BaseLayer):
    def __init__(self, input, padding = "SAME", name="emsamble_mean"):
        BaseLayer.__init__(self, input = input, name=name, type = "emsamble_mean")
        
    
    def build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.output = tf.reduce_mean(input, axis = -1)
        
        
        
        
        
        