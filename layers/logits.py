from base.baselayer import BaseLayer
import tensorflow as tf


class Logits(BaseLayer):
    def __init__(self, input, num_classes, name="fc"):
        self._num_classes = num_classes
        BaseLayer.__init__(self, input = input, name=name, type = "fc_conv")
        
    
    def _build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        kernel = input.shape.as_list()[1:-1]
        self.output = tf.squeeze(self.conv2d_fc(input,  self._num_classes , kernel, name = self.name), [1,2])
#         input = self.input.output if isinstance(self.input, BaseLayer) else self.input
#         self.output = self.fully_connected(input = input, 
#                                            hidden_units = self._num_classes, 
#                                            keep = None, activation = None, 
#                                            name = self.name, 
#                                            batch_norm = None)
        
        
        
        
        
        