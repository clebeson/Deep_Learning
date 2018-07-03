from base.baselayer import BaseLayer



class FullyConnected(BaseLayer):
    def __init__(self,input, hidden_units, name="fc", keep = None, activation = "relu", norm = None, istraining = True):
        self._hidden_units = hidden_units
        self._keep = keep
        self._norm = norm
        self._activation = activation
        BaseLayer.__init__(self, input = input, name=name, type = "fc_conv", istraining = istraining)
        
    
    def _build(self):
        input = self.input.output if isinstance(self.input, BaseLayer) else self.input

        rank = len(input.shape.as_list())
        if rank == 4: 
            kernel = input.shape.as_list()[1:-1]
            self.conv2d_fc(input,  self._hidden_units , kernel,  activation = self._activation, keep = self._keep, name = self.name, batch_norm = self._norm, padding = "VALID")
        else:
            self.type ="fc"
            self.output = self.fully_connected(input = input, 
                                            hidden_units = self._hidden_units, 
                                            keep = self._keep, 
                                            activation = self._activation, 
                                            name = self.name, 
                                            batch_norm = self._norm)
        
        
        