from base.baselayer import BaseLayer



class Conv2d(BaseLayer):
    def __init__(self, input, out_size, kernel = [3,3], norm = None, padding = "SAME", name="conv/conv"):
        self._out_size = out_size
        self._norm = norm
        self._padding = padding
        self._kernel = kernel
        BaseLayer.__init__(self, input = input, name=name, type = "conv2d")
        
    
    def _build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input
        self.conv2d(input, self._out_size, self._kernel,  name = self.name,  batch_norm = self._norm, padding = self._padding)
        
        
        
        
        
        