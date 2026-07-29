import hydra
from omegaconf import DictConfig

from spnn.training.training_kfold import training_KFold

class Args:
    """ a Struct Class  """
    pass
args=Args()
args.config_name='SPNN.yaml'
@hydra.main(config_path='./conf/SPNN/', config_name=args.config_name, version_base='1.1')
def main(cfg: DictConfig):
    training_KFold(cfg,args)

if __name__ == "__main__":
    main()
