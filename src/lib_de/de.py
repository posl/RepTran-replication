import pickle
from .searcher import Searcher
import torch
import numpy as np
from tqdm import tqdm

from logging import getLogger

logger = getLogger("base_logger")

class DE_searcher(Searcher):
    random = __import__("random")

    importlib = __import__("importlib")
    base = importlib.import_module("deap.base")
    creator = importlib.import_module("deap.creator")
    tools = importlib.import_module("deap.tools")
    logger.info("Initializing DE_searcher...")

    def __init__(
        self,
        # inputs,
        # labels,
        # indices_to_correct,
        # indices_to_wrong,
        # num_label,
        # indices_to_target_layers,
        # task_name,
        # device,
        # mutation=(0.5, 1),
        # recombination=0.7,
        # max_search_num=100,
        # model=None,
        # patch_aggr=None,
        # batch_size=None,
        # is_lstm=False,
        # is_multi_label=True,
        # len_for_repair=None,
    ):
        """ """
        # super(DE_searcher, self).__init__(
        #     inputs,
        #     labels,
        #     indices_to_correct,
        #     indices_to_wrong,
        #     num_label,
        #     indices_to_target_layers,
        #     task_name,
        #     device,
        #     max_search_num=max_search_num,
        #     model=model,
        #     batch_size=batch_size,
        #     is_lstm=is_lstm,
        #     is_multi_label=is_multi_label,
        #     len_for_repair=len_for_repair,
        # )

        # fitness computation related initialisation
        self.fitness = 0.0

        # deap関連の設定
        self.creator.create("FitnessMax", self.base.Fitness, weights=(1.0,))  # maximisation
        self.creator.create("Individual", np.ndarray, fitness=self.creator.FitnessMax, model_name=None)

        # store the best performace seen so far
        self.the_best_performance = None
        # self.max_num_of_unchanged = int(self.max_search_num / 10) if self.max_search_num is not None else None

        # if self.max_num_of_unchanged < 10:
        #     self.max_num_of_unchanged = 10

        self.num_iter_unchanged = 0

        # DE specific
        # self.mutation = mutation
        # self.recombination = recombination

        # # fitness
        # self.patch_aggr = patch_aggr

        logger.info("Finish Initializing DE_searcher...")