from spnn.models.spdnet import SPDNet
from spnn.models.spdresnet import RResNet
from spnn.models.spnn import SPNN
from spnn.models.spdnetrbn import SPDNetRBN
from spnn.models.gyrospd import GyroSPD
from spnn.models.gyrospdpp import GyroSPDpp

classes = {
    "SPDNet": SPDNet,
    "SPDResNet": RResNet,
    "SPNN": SPNN,
    "SPDNetRBN": SPDNetRBN,
    "GyroSPD": GyroSPD,
    "GyroSPDpp": GyroSPDpp,
}

def get_model(args):
    model = classes[args.model_type](args)
    return model
