from ..api.topology import DMFFTopology


class BaseOperator:
    def __init__(self, ffinfo):
        pass

    def __call__(self, input: DMFFTopology, **kwargs) -> DMFFTopology:
        return self.operate(input, **kwargs)
        
    def operate(self, topdata: DMFFTopology) -> DMFFTopology:
        pass
