from base.baselayer import BaseLayer
import tensorflow as tf


class Logits(BaseLayer):
    def __init__(self, input, num_classes, name="fc"):
        self._num_classes = num_classes
        BaseLayer.__init__(self, input = input, name=name, type = "fc_conv")
        
    
    def build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        rank = len(input.shape.as_list())
        if rank == 4: 
            kernel = input.shape.as_list()[1:-1]
            self.output = tf.squeeze(self.conv2d_fc(input,  self._num_classes , kernel, name = self.name), [1,2])
        else:
            self.type="fc"
            self.output = self.fully_connected(input = input, 
                                               hidden_units = self._num_classes, 
                                               keep = None, activation = None, 
                                               name = self.name, 
                                               batch_norm = None)
        
        
        
        
        
        