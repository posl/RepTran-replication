import pickle, os, sys, time, math
# utilsをインポートできるようにパスを追加
import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from utils.data_util import format_label

from logging import getLogger

logger = getLogger("base_logger")

class DE_searcher(object):
    random = __import__("random")
    importlib = __import__("importlib")
    base = importlib.import_module("deap.base")
    creator = importlib.import_module("deap.creator")
    tools = importlib.import_module("deap.tools")
    logger.info("Initializing DE_searcher...")

    def __init__(
        self,
        inputs,
        labels,
        indices_to_correct,
        indices_to_wrong,
        num_label,
        indices_to_target_layers,
        device,
        tgt_vdiff,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=100,
        model=None,
        patch_aggr=None,
        batch_size=None,
        is_multi_label=True,
    ):
        super(DE_searcher, self).__init__()
        self.device = device
        self.inputs = inputs
        self.indices_to_correct = indices_to_correct
        self.indices_to_wrong = indices_to_wrong
        self.num_label = num_label
        self.indices_to_target_layers = indices_to_target_layers
        self.targeted_layer_names = None
        self.tgt_vdiff = tgt_vdiff
        self.batch_size = batch_size
        self.is_multi_label = is_multi_label
        self.maximum_fitness = 0.0  # the maximum fitness value
        self.mdl = model
        self.max_search_num = max_search_num
        self.indices_to_sampled_correct = None
        self.tgt_pos = 0 # TODO: should not be hard coded

        # ラベルの設定
        if is_multi_label:
            # ラベルが1次元配列になっているのでonehotベクトル化(2次元になる)
            if not isinstance(labels[0], Iterable):
                self.ground_truth_labels = labels
                self.labels = format_label(labels, self.num_label)
            # ラベルは2次元配列
            else:
                self.labels = labels
                self.ground_truth_labels = np.argmax(self.labels, axis=1)
        else:
            self.labels = labels
            self.ground_truth_labels = labels

        # fitness computation related initialisation
        self.fitness = 0.0

        # deap関連の設定
        self.creator.create("FitnessMax", self.base.Fitness, weights=(1.0,))  # maximisation
        self.creator.create("Individual", np.ndarray, fitness=self.creator.FitnessMax, model_name=None)

        # store the best performace seen so far
        self.the_best_performance = None
        self.max_num_of_unchanged = int(self.max_search_num / 10) if self.max_search_num is not None else None

        if self.max_num_of_unchanged < 10:
            self.max_num_of_unchanged = 10

        self.num_iter_unchanged = 0

        # DE specific
        self.mutation = mutation
        self.recombination = recombination

        # fitness
        self.patch_aggr = patch_aggr

        logger.info("Finish Initializing DE_searcher...")

    def eval(self, patch_candidate, places_to_fix):
        # self.inputsのデータを予測してlossを使ったfitness functionの値を返す
        # データの予測時は，places_to_fixの位置のニューロンをpatch_candidateの値に変更して予測する

        all_proba = []
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none") # NOTE: バッチ内の各サンプルずつのロスを出すため. デフォルトのreduction="mean"だとバッチ内の平均になってしまう
        # batchごとに取り出して予測実行
        for data_idx, entry_dic in tqdm(enumerate(self.inputs.iter(batch_size=self.batch_size)), 
                                    total=math.ceil(len(self.inputs)/self.batch_size)): # NOTE: shuffleされない
            x, y = entry_dic["pixel_values"].to(self.device), np.array(entry_dic["labels"])
            # imp_posはレイヤ番号とニューロン番号のリスト
            assert len(patch_candidate) == len(places_to_fix), f"len(patch_candidate) must be equal to len(places_to_fix), but got {len(patch_candidate)} and {len(places_to_fix)}"
            # バッチに対応するhidden statesとintermediate statesの取得
            outputs = self.mdl(x, tgt_pos=self.tgt_pos, imp_pos=places_to_fix, imp_op=patch_candidate)
            # outputs.logitsを確率にする
            proba = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # sampleごとのロスを計算
            loss = loss_fn(proba, torch.from_numpy(y).to(self.device)).cpu().detach().numpy()
            all_proba.append(proba.detach().cpu().numpy())
            losses_of_all.append(loss)
        losses_of_all = np.concatenate(losses_of_all, axis=0) # (num_of_data, )
        all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
        all_pred_laebls = np.argmax(all_proba, axis=-1) # (num_of_data, )
        # 予測結果が合ってるかどうかを評価
        is_correct = all_pred_laebls == self.ground_truth_labels
        # 元々正解だったサンプルが変わらず正解だった数を取得
        num_intact = sum(is_correct[self.indices_to_correct])
        # 元々不正解だったサンプルが正解になった数を取得
        num_patched = sum(is_correct[self.indices_to_wrong])
        # 元々正解だったサンプルに対するロスの平均を取得
        losses_of_correct = np.mean(losses_of_all[self.indices_to_correct])
        # 元々不正解だったサンプルに対するロスの平均を取得
        losses_of_wrong = np.mean(losses_of_all[self.indices_to_wrong])

        fitness_for_correct = (num_intact / len(self.indices_to_correct) + 1) / (losses_of_correct + 1)
        fitness_for_wrong = (num_patched / len(self.indices_to_wrong) + 1) / (losses_of_wrong + 1)
        final_fitness = fitness_for_correct + fitness_for_wrong
        return (final_fitness,)


    def search(self, places_to_fix, save_path):

        # ターゲットとなるレイヤ達の重みの平均や分散を取得
        num_places_to_fix = len(places_to_fix)  # the number of places to fix # NDIM of a patch candidate

        # set search parameters
        pop_size = 100
        toolbox = self.base.Toolbox()

        # 初期値はターゲットレイヤの重みの平均と分散を用いて正規分布からサンプリング (len(places_to_fix)個をサンプリング)
        def init_indiv():
            v_sample = lambda tgt_vdiff: np.random.normal(loc=tgt_vdiff, scale=1, size=1)[0]
            ind = np.float32(list(map(v_sample, self.tgt_vdiff)))
            return ind

        # DEのためのパラメータ設定
        toolbox.register("individual", self.tools.initIterate, self.creator.Individual, init_indiv)
        toolbox.register("population", self.tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", np.random.choice, size=3, replace=False)
        toolbox.register("evaluate", self.eval)

        # set logbook
        stats = self.tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = self.tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields
        # logbook setting end

        # ここで，長さplaces_to_fixの初期値ベクトルをpop_size個生成
        pop = toolbox.population(n=pop_size)

        hof = self.tools.HallOfFame(1, similar=np.array_equal)

        # update fitness
        # print ("Places to fix", places_to_fix)

        # 各個体のfitnessを計算
        logger.info("Evaluating initial population...")
        for ind in tqdm(pop, total=len(pop), desc=f"processing initial population"):
            ind.fitness.values = toolbox.evaluate(ind, places_to_fix)
            ind.model_name = None

        # 初期のベスト（暫定）を更新
        hof.update(pop)
        best = hof[0]
        best.model_name = "initial"
        logger.info(f"Initial fitness: {best.fitness.values[0]} at X_best: {best}, model_name: {best.model_name}")

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)

        search_start_time = time.time()
        # search start
        import time

        # 世代の繰り返し
        for iter_idx in range(self.max_search_num):
            iter_start_time = time.time()
            MU = self.random.uniform(self.mutation[0], self.mutation[1])

            # 各個体について繰り返し
            for pop_idx, ind in tqdm(enumerate(pop), total=len(pop), desc=f"iter_idx: {iter_idx}"):
                # set model name
                new_model_name = "iter{}-pop{}".format(iter_idx, pop_idx)

                # select
                target_indices = [_i for _i in range(pop_size) if _i != pop_idx]
                a_idx, b_idx, c_idx = toolbox.select(target_indices)
                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]

                y = toolbox.clone(ind)

                index = self.random.randrange(num_places_to_fix)
                for i, value in enumerate(ind):
                    # 一部を変異させる
                    if i == index or self.random.random() < self.recombination:
                        # パッチ候補の値は対象レイヤの重みから計算したバウンドに収める
                        # y[i] = np.clip(a[i] + MU * (b[i] - c[i]), bounds[i][0], bounds[i][1])
                        y[i] = a[i] + MU * (b[i] - c[i])

                y.fitness.values = toolbox.evaluate(y, places_to_fix)
                if y.fitness.values[0] >= ind.fitness.values[0]:  # better
                    pop[pop_idx] = y  # upddate
                    # set new model name
                    pop[pop_idx].model_name = new_model_name
                    # update best
                    if best.fitness.values[0] < pop[pop_idx].fitness.values[0]:
                        hof.update(pop)
                        best = hof[0]
                        # print ("New best one is set: {},
                        # fitness: {}, model_name: {}".format(
                        # best, best.fitness.values[0], best.model_name))

            # 全population終わったらその世代でのベストのパッチ候補を表示
            hof.update(pop)
            best = hof[0]
            logger.info(
                f"[The best at Gen {iter_idx}] fitness={best.fitness.values[0]} at X_best={best}, model_name: {best.model_name}"
            )

            # logging for this generation
            record = stats.compile(pop)
            logbook.record(gen=iter_idx, evals=len(pop), **record)

            #########################################################
            # update for best value to check for early stop #########
            #########################################################
            # レイヤのインデックスをキー，そのレイヤの修正後の重みを値とする辞書
            deltas = {}  # this is deltas for set update op
            for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
                if idx_to_tl not in deltas.keys():
                    deltas[idx_to_tl] = self.init_weights[idx_to_tl]
                # since our op is set
                deltas[idx_to_tl][tuple(inner_indices)] = best[i]

            # check for two stop coniditions
            # ここでearly stopの判定を行う (fitnessが変化しないエポックが一定数続いたら終了)
            if self.is_the_performance_unchanged(best):
                logger.info("Performance has not been changed over {} iterations".format(self.num_iter_unchanged))
                break

        # with these two cases, the new model has not been saved
        # if self.empty_graph is not None:
        logger.info(f"best ind.: {best}, fitness: {best.fitness.values[0]}")

        # 重みの辞書更新
        deltas = {}  # this is deltas for set update op
        for i, (idx_to_tl, inner_indices) in enumerate(places_to_fix):
            if idx_to_tl not in deltas.keys():
                deltas[idx_to_tl] = self.init_weights[idx_to_tl]
            # since our op is set
            deltas[idx_to_tl][tuple(inner_indices)] = best[i]
        # pklに保存
        with open(save_path, "wb") as f:
            pickle.dump(deltas, f)
        logger.info("The model is saved to {}".format(save_path))
