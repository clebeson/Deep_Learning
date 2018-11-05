from base.baselayer import BaseLayer



class Conv3d(BaseLayer):
    def __init__(self, input, temp_filter_size, num_filters, kernel = [3,3],  stride=[1,1,1,1,1], norm = None, padding = "SAME", name="con3d/conv", istraining = True):
        self._temp_filter_size = temp_filter_size
        self._num_filters = num_filters
        self._norm = norm
        self._stride = stride
        self._padding = padding
        self._kernel = kernel
        BaseLayer.__init__(self, input = input, name=name, type = "conv3d")
        
    
    def build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.output = self.conv3d(tensor = input, temp_kernel_size = self._temp_filter_size, 
                    spatial_kernel = self._kernel, num_filters = self._num_filters, 
                    stride=self._stride, batch_norm = self._norm, name = self.name)
        
        
        
        
        
        