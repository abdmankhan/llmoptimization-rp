from optimizers.random_search import random_search
from optimizers.nsga2 import nsga2_optimize
from optimizers.spea2 import spea2_optimize
from optimizers.mopso import mopso_optimize
# from optimizers.moead import moead_optimize
# from optimizers.rnsgaii import rnsgaii_optimize
# from optimizers.moead_gen import moead_gen_optimize


OPTIMIZERS = [
    {
        "name": "Random Search",
        "fn": random_search
    },
    {
        "name": "NSGA-II",
        "fn": nsga2_optimize
    },
    {
        "name": "SPEA2",
        "fn": spea2_optimize
    },
    # {
    #     "name": "MOPSO",
    #     "fn": mopso_optimize
    # },
    # {
    #     "name": "MOEA/D",
    #     "fn": moead_optimize
    # },
    # {
    #     "name": "RNSGA-II",
    #     "fn": rnsgaii_optimize
    # },
    # {
    #     "name": "MOEA/D-GEN",
    #     "fn": moead_gen_optimize
    # }
]
