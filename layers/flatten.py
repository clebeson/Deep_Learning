from base.baselayer import BaseLayer
import tensorflow as tf



class Flatten(BaseLayer):
    def __init__(self, input, name="flatten"):
        BaseLayer.__init__(self, input = input, name=name, type = "flatten")
        
    
    def build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.output = tf.layers.flatten(input, name = self.name)
        
        
        
        
        
        