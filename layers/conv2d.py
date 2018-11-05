from base.baselayer import BaseLayer



class Conv2d(BaseLayer):
    def __init__(self, input, out_size, kernel = [3,3], stride = [1,1,1,1], norm = None, padding = "SAME", name="conv/conv", istraining = True):
        BaseLayer.__init__(self, input = input, name=name, type = "conv2d")

        self._out_size = out_size
        self.stride = stride
        self._norm = norm
        self._padding = padding
        self._kernel = kernel
        self._istraining_placeholder = istraining

        
    
    def build(self):
        

        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.output = self.conv2d(input, self._out_size, self._kernel, stride = self.stride, name = self.name,  batch_norm = self._norm, padding = self._padding)
        
        
        
        
        
        