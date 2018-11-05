from base.baselayer import BaseLayer
from roi_pooling.roi_pooling_ops import roi_pooling



class ROIPool(BaseLayer):
    def __init__(self, input, kernel_size = 2, stride = 2, depth = 1, name="pool"):
        self._kernel = kernel_size
        self._stride = stride
        self._depth = depth
        BaseLayer.__init__(self, input = input, name=name, type = "maxpool")
        
    
    def build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self. output = self.maxpool(input, 
                                    k=self._kernel, 
                                    d= self._depth , 
                                    stride = self._stride, 
                                    name = self.name)
        
        
        
        
        