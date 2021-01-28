from ...classifiers import TSN
from ....builder import MODELS


@MODELS.register_module()
class SpeedNet(TSN):

    def __init__(self, *args, **kwargs):
        super(SpeedNet, self).__init__(*args, **kwargs)
