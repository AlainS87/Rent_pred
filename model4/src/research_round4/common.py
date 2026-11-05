from pathlib import Path
from src.backtest.protocol import ROOT
from src.research_round2.models import expected_contract
from src.research_round2 import models as b0_models
from src.research_round3 import models as c3_models

from configs import config_path
CONFIG = config_path('research_round4/protocol.json')


def recipe(config,name):
    return config['model_recipes'][name]


def model_contract(history,query,origin,name,config,pin):
    value=expected_contract(history,query,origin,name,recipe(config,name),pin)
    value['smoke_iterations']=None
    return value


def restore_model(directory,contract,name):
    return (b0_models if name=='B0' else c3_models).restore(Path(directory),contract)
