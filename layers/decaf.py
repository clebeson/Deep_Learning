from base.baselayer import BaseLayer
import tensorflow as tf



class Decaf(BaseLayer):
    def __init__(self, input, name="Decaf"):
        BaseLayer.__init__(self, input = input, name=name, type = "Decaf", istraining = False)
        
    
    def _build(self):
        output_layers = []
        for layer_in in self.input:
            input = layer_in.output if isinstance(layer_in, BaseLayer) else layer_in
            output_layers.append(tf.layers.flatten(input, name = self.name))
        self.output = tf.concat(output_layers, 1)
        
        
        
        
        
        