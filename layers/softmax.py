from base.baselayer import BaseLayer
import tensorflow as tf


class Softmax(BaseLayer):
    def __init__(self, input, name="softmax"):

        BaseLayer.__init__(self, input = input, name=name, type = "softmax", istraining = False)
        
    
    def _build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.output = tf.layers.fatten(input, name = self.name)
        
        
        
        
        