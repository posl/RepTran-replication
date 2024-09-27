import pickle, os, sys, time, math
# utilsをインポートできるようにパスを追加
import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from utils.data_util import format_label, make_batch_of_label

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
        batch_hs_before_layernorm,
        labels,
        indices_to_correct,
        indices_to_wrong,
        num_label,
        indices_to_target_layers,
        device,
        tgt_vdiff=None,
        mutation=(0.5, 1),
        recombination=0.7,
        max_search_num=100,
        partial_model=None,
        alpha=None,
        batch_size=None,
        is_multi_label=True,
        pop_size=50,
        mode="neuron",
        weight_before2med=None,
        weight_med2after=None,
        pos_before=None,
        pos_after=None,
    ):
        super(DE_searcher, self).__init__()
        self.device = device
        self.batch_hs_before_layernorm = batch_hs_before_layernorm
        self.indices_to_correct = indices_to_correct
        self.indices_to_wrong = indices_to_wrong
        self.num_label = num_label
        self.indices_to_target_layers = indices_to_target_layers
        self.targeted_layer_names = None
        self.tgt_vdiff = tgt_vdiff
        self.batch_size = batch_size
        self.is_multi_label = is_multi_label
        self.maximum_fitness = 0.0  # the maximum fitness value
        self.mdl = partial_model
        self.max_search_num = max_search_num
        self.indices_to_sampled_correct = None
        self.tgt_pos = 0 # TODO: should not be hard coded
        self.pop_size = pop_size
        self.mode = mode

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
        self.batched_labels = make_batch_of_label(labels, batch_size)

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
        self.alpha = alpha

        # modeがweightの場合の処理
        if mode == "weight":
            self.weight_before2med = weight_before2med
            self.weight_med2after = weight_med2after
            logger.info(f"self.weight_before2med.shape: {self.weight_before2med.shape}")
            logger.info(f"self.weight_med2after.shape: {self.weight_med2after.shape}")
            # DEの初期値生成のために平均と標準偏差を出しておく
            self.mean_b2m = np.mean(self.weight_before2med)
            self.std_b2m = np.std(self.weight_before2med)
            self.mean_m2a = np.mean(self.weight_med2after)
            self.std_m2a = np.std(self.weight_med2after)
            # pos_before, afterの長さを取得しておく
            self.num_pos_before = len(pos_before)
            self.num_pos_after = len(pos_after)
            self.num_total_pos = self.num_pos_before + self.num_pos_after
        
        # modeがneuronでもweightでもない場合はエラー終了
        if mode not in ["neuron", "weight"]:
            NotImplementedError("mode should be either 'neuron' or 'weight'")

        logger.info("Finish Initializing DE_searcher...")

    def eval_neurons(self, patch_candidate, places_to_fix, show_log=True):
        # self.inputsのデータを予測してlossを使ったfitness functionの値を返す
        # データの予測時は，places_to_fixの位置のニューロンをpatch_candidateの値に変更して予測する
        all_proba = []
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none") # NOTE: バッチ内の各サンプルずつのロスを出すため. デフォルトのreduction="mean"だとバッチ内の平均になってしまう
        for cached_state, y in zip(self.batch_hs_before_layernorm, self.batched_labels):
            logits = self.mdl(hidden_states_before_layernorm=cached_state, tgt_pos=self.tgt_pos, imp_pos=places_to_fix, imp_op=patch_candidate)
            # 出力されたlogitsを確率に変換
            proba = torch.nn.functional.softmax(logits, dim=-1)
            all_proba.append(proba.detach().cpu().numpy())
            # sampleごとのロスを計算
            loss = loss_fn(proba, torch.from_numpy(y).to(self.device)).cpu().detach().numpy()
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
        final_fitness = self.alpha * fitness_for_correct + fitness_for_wrong

        if show_log:
            logger.info(f"num_intact: {num_intact}/{len(self.indices_to_correct)}, num_patched: {num_patched}/{len(self.indices_to_wrong)}")
        return (final_fitness,)

    def eval_weights(self, patch_candidate, pos_before, pos_after, show_log=True):
        assert len(patch_candidate) == self.num_total_pos, "The length of patch_candidate should be equal to the number of total positions"
        # patch_candidateの最初のself.num_pos_before個はbefore2medの重みの修正，その後のself.num_pos_after個はmed2afterの重みの修正
        for ba, pos in enumerate([pos_before, pos_after]):
            # patch_candidateのindexを設定
            if ba == 0:
                idx_patch_candidate = range(0, self.num_pos_before)
                assert len(idx_patch_candidate) == self.num_pos_before, "The length of idx_patch_candidate should be equal to the number of positions before"
                tgt_weight_data = self.mdl.base_model_last_layer.intermediate.dense.weight.data
            else:
                idx_patch_candidate = range(self.num_pos_before, self.num_total_pos)
                assert len(idx_patch_candidate) == self.num_pos_after, "The length of idx_patch_candidate should be equal to the number of positions after"
                tgt_weight_data = self.mdl.base_model_last_layer.output.dense.weight.data
            # posで指定された位置のニューロンを書き換える
            xi, yi = pos[:, 0], pos[:, 1]
            tgt_weight_data[xi, yi] = torch.from_numpy(patch_candidate[idx_patch_candidate]).to(self.device)
        
        # self.inputsのデータを予測してlossを使ったfitness functionの値を返す
        all_proba = []
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        for cached_state, y in zip(self.batch_hs_before_layernorm, self.batched_labels):
            logits = self.mdl(hidden_states_before_layernorm=cached_state, tgt_pos=self.tgt_pos)
            # 出力されたlogitsを確率に変換
            proba = torch.nn.functional.softmax(logits, dim=-1)
            all_proba.append(proba.detach().cpu().numpy())
            # sampleごとのロスを計算
            loss = loss_fn(proba, torch.from_numpy(y).to(self.device)).cpu().detach().numpy()
            losses_of_all.append(loss)
        losses_of_all = np.concatenate(losses_of_all, axis=0) # (num_of_data, )
        losses_to_correct = losses_of_all[self.indices_to_correct]
        losses_to_wrong = losses_of_all[self.indices_to_wrong]
        all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
        all_pred_labels = np.argmax(all_proba, axis=-1) # (num_of_data, )
        # 予測結果が合ってるかどうかを評価
        is_correct = all_pred_labels == self.ground_truth_labels # 修正後モデルで正しい予測のデータインデックス
        indices_to_correct_to_correct = is_correct[self.indices_to_correct] # 元々正解だったサンプルが変わらず正解だったサンプルのインデックス
        indices_to_correct_to_wrong = ~indices_to_correct_to_correct # 元々正解だったサンプルが不正解に変わったサンプルのインデックス
        indices_to_wrong_to_correct = is_correct[self.indices_to_wrong] # 元々不正解だったサンプルが正解に変わったサンプルのインデックス
        indices_to_wrong_to_wrong = ~indices_to_wrong_to_correct # 元々不正解だったサンプルが変わらず不正解だったサンプルのインデックス
        assert sum(indices_to_correct_to_correct) + sum(indices_to_correct_to_wrong) == len(self.indices_to_correct), f"The sum of indices_to_correct_to_correct and indices_to_correct_to_wrong should be equal to the length of indices_to_correct (sum(indices_to_correct_to_correct): {sum(indices_to_correct_to_correct)}, sum(indices_to_correct_to_wrong): {sum(indices_to_correct_to_wrong)}, len(self.indices_to_correct): {len(self.indices_to_correct)}"
        assert sum(indices_to_wrong_to_correct) + sum(indices_to_wrong_to_wrong) == len(self.indices_to_wrong), f"The sum of indices_to_wrong_to_correct and indices_to_wrong_to_wrong should be equal to the length of indices_to_wrong (sum(indices_to_wrong_to_correct): {sum(indices_to_wrong_to_correct)}, sum(indices_to_wrong_to_wrong): {sum(indices_to_wrong_to_wrong)}, len(self.indices_to_wrong): {len(self.indices_to_wrong)}"

        # TODO: fitness_fnの形は微妙にバリエーションがあるのでカスタマイズできるようにしたい．
        # 元々正解だったサンプルが変わらず正解だった数を取得
        num_intact = sum(indices_to_correct_to_correct)
        # 元々不正解だったサンプルが正解になった数を取得
        num_patched = sum(indices_to_wrong_to_correct)
        # 元々正解だったサンプルに対するロスの平均を取得
        mean_of_losses_of_correct = np.mean(losses_of_all[self.indices_to_correct])
        # 元々不正解だったサンプルに対するロスの平均を取得
        mean_of_losses_of_wrong = np.mean(losses_of_all[self.indices_to_wrong])
        # 元々正解だったサンプルのうち不正解に変わってしまったサンプルに対するロス
        losses_of_correct_to_wrong = losses_to_correct[indices_to_correct_to_wrong]
        # 元々不正解だったサンプルのうち変わらず不正解だったサンプルに対するロス
        losses_of_wrong_to_wrong = losses_to_wrong[indices_to_wrong_to_wrong]

        # fitness_for_correct = (num_intact / len(self.indices_to_correct) + 1) / (mean_of_losses_of_correct + 1)
        # fitness_for_wrong = (num_patched / len(self.indices_to_wrong) + 1) / (mean_of_losses_of_wrong + 1)
        # final_fitness = self.alpha * fitness_for_correct + fitness_for_wrong
        # terms for correct

        # below is the same as Arachne-v2
        term1_pos = num_intact
        term2_pos = np.sum(1 / (losses_of_correct_to_wrong + 1)) if len(losses_of_correct_to_wrong) > 0 else 0
        fitness_for_correct = (term1_pos + term2_pos) / len(self.indices_to_correct)
        term1_neg = num_patched
        term2_neg = np.sum(1 / (losses_of_wrong_to_wrong + 1)) if len(losses_of_wrong_to_wrong) > 0 else 0
        fitness_for_wrong = (term1_neg + term2_neg) / len(self.indices_to_wrong)
        # fitness_for_correct, fitness_for_wrongはどちらも[0, 1]の範囲に収まる
        assert 0 <= fitness_for_correct <= 1, f"fitness_for_correct should be in [0, 1] (fitness_for_correct: {fitness_for_correct})"
        assert 0 <= fitness_for_wrong <= 1, f"fitness_for_wrong should be in [0, 1] (fitness_for_wrong: {fitness_for_wrong})"
        # print(f"num_intact: {num_intact}/{len(self.indices_to_correct)}, num_patched: {num_patched}/{len(self.indices_to_wrong)}")
        # print(f"fitness_for_correct: {fitness_for_correct}, fitness_for_wrong: {fitness_for_wrong}")
        final_fitness = (1-self.alpha) * fitness_for_correct + self.alpha *  fitness_for_wrong
        # 思い切ってintact_rateとpatched_rateだけにしちゃう
        # final_fitness = num_intact / len(self.indices_to_correct) + self.alpha * num_patched / len(self.indices_to_wrong)
        
        if show_log:
            logger.info(f"num_intact: {num_intact}/{len(self.indices_to_correct)} ({100*num_intact/len(self.indices_to_correct):.2f}%), num_patched: {num_patched}/{len(self.indices_to_wrong)} ({100*num_patched/len(self.indices_to_wrong):.2f}%)")
        tracking_dict = {
            "fitness": final_fitness,
            "fitness_for_correct": fitness_for_correct,
            "fitness_for_wrong": fitness_for_wrong,
            "term1_pos": term1_pos,
            "term2_pos": term2_pos,
            "term1_neg": term1_neg,
            "term2_neg": term2_neg,
            "all_pred_labels": all_pred_labels,
        }
        return final_fitness, tracking_dict

    def is_the_performance_unchanged(self, curr_best_patch_candidate):
        """
        curr_best_performance: fitness of curr_best_patch_candidate:
        Ret (bool):
                True: the performance
        """
        curr_best_performance = curr_best_patch_candidate.fitness.values[0]

        if self.the_best_performance is None:
            self.the_best_performance = curr_best_performance
            return False

        if np.float32(curr_best_performance) == np.float32(self.the_best_performance):
            self.num_iter_unchanged += 1
        else:
            self.num_iter_unchanged = 0  # look for subsequent
            if curr_best_performance > self.the_best_performance:
                self.the_best_performance = curr_best_performance

        if self.max_num_of_unchanged < self.num_iter_unchanged:
            return True
        else:
            return False

    def search(self, patch_save_path, places_to_fix=None, pos_before=None, pos_after=None, tracker_save_path=None):
        # set search parameters
        pop_size = self.pop_size
        toolbox = self.base.Toolbox()

        # DEの経過観察用のデータ
        fitness_tracker = {
            "fitness": [],
            "fitness_for_correct": [],
            "fitness_for_wrong": [],
            "term1_pos": [],
            "term2_pos": [],
            "term1_neg": [],
            "term2_neg": [],
            "all_pred_labels": [],
        }

        # ニューロン修正の際の初期値生成 (ニューロンのx倍なので，初期値はN(1, 1)からサンプリング)
        def init_indiv_neurons():
            v_sample = lambda mean_v: np.random.normal(loc=mean_v, scale=1, size=1)[0] # N(mean_v, 1)からサンプリング
            ind = np.float32(list(map(v_sample, np.ones(num_places_to_fix, dtype=np.float32))))
            return ind
        
        # 初期値はターゲットレイヤの重みの平均と分散を用いて正規分布からサンプリング (len(places_to_fix)個をサンプリング)
        def init_indiv_weights():
            v_sample = lambda mean_v, std_v: np.random.normal(loc=mean_v, scale=std_v, size=1)[0] # N(mean_v, std_v)からサンプリング
            ind = np.float32(list(map(v_sample, self.mean_values, self.std_values)))
            return ind
        
        # DEのpop初期化関数を設定
        if self.mode == "neuron":
            assert places_to_fix is not None, "places_to_fix should be set"
            args_for_eval = (places_to_fix,)
            # ターゲットとなるレイヤ達の重みの平均や分散を取得
            num_places_to_fix = len(places_to_fix)  # the number of places to fix # NDIM of a patch candidate
            init_indiv = init_indiv_neurons
            eval_func = self.eval_neurons
        elif self.mode == "weight":
            assert pos_before is not None, "pos_before should be set"
            assert pos_after is not None, "pos_after should be set"
            args_for_eval = (pos_before, pos_after)
            self.mean_values = []
            self.std_values = []
            # self.mean_valuesとself.std_valuesはそれぞれ，最初のself.num_pos_before個についてはself.mean_b2m, その後のself.num_pos_after個についてはself.mean_m2a のようにする (標準偏差も同様)
            for i in range(self.num_total_pos):
                if i < self.num_pos_before:
                    self.mean_values.append(self.mean_b2m)
                    self.std_values.append(self.std_b2m)
                else:
                    self.mean_values.append(self.mean_m2a)
                    self.std_values.append(self.std_m2a)
            num_places_to_fix = self.num_total_pos
            init_indiv = init_indiv_weights
            eval_func = self.eval_weights

        # DEのためのパラメータ設定
        toolbox.register("individual", self.tools.initIterate, self.creator.Individual, init_indiv)
        toolbox.register("population", self.tools.initRepeat, list, toolbox.individual)
        toolbox.register("select", np.random.choice, size=3, replace=False)
        toolbox.register("evaluate", eval_func)

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
            eval_ret = toolbox.evaluate(ind, *args_for_eval)
            ind.fitness.values = (eval_ret[0], )
            ind.tracking_dict = eval_ret[1]
            ind.model_name = None

        # 初期のベスト（暫定）を更新
        hof.update(pop)
        best = hof[0]
        best.model_name = "initial"
        logger.info(f"Initial fitness: {best.fitness.values[0]} at X_best: {best}, model_name: {best.model_name}")

        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(pop), **record)

        # search start
        search_start_time = time.perf_counter()

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

                eval_ret = toolbox.evaluate(ind, *args_for_eval)
                y.fitness.values = (eval_ret[0], )
                y.tracking_dict = eval_ret[1]
                logger.info(f"[{new_model_name}] fitness: {y.fitness.values[0]}")
                if y.fitness.values[0] >= ind.fitness.values[0]:  # better
                    pop[pop_idx] = y  # upddate
                    # set new model name
                    pop[pop_idx].model_name = new_model_name
                    # update best
                    if best.fitness.values[0] < pop[pop_idx].fitness.values[0]:
                        hof.update(pop)
                        best = hof[0]
                        logger.info(f"New best one is set: fitness: {best.fitness.values[0]}, model_name: {best.model_name}")

            # 全population終わったらその世代でのベストのパッチ候補を表示
            hof.update(pop)
            best = hof[0]
            logger.info(
                f"[The best at Gen {iter_idx}] fitness={best.fitness.values[0]} at X_best={best}, model_name: {best.model_name}"
            )
            # その時点でのbestのfitnessなどを更新する
            for key, value in best.tracking_dict.items():
                fitness_tracker[key].append(value)
            logger.info(f"fitness_tracker: {fitness_tracker}")

            # logging for this generation
            record = stats.compile(pop)
            logbook.record(gen=iter_idx, evals=len(pop), **record)

            # check for two stop coniditions
            # ここでearly stopの判定を行う (fitnessが変化しないエポックが一定数続いたら終了)
            if self.is_the_performance_unchanged(best):
                logger.info("Performance has not been changed over {} iterations".format(self.num_iter_unchanged))
                break

        # with these two cases, the new model has not been saved
        # if self.empty_graph is not None:
        logger.info(f"best ind.: {best}, fitness: {best.fitness.values[0]}")

        # bestをnpyで保存
        np.save(patch_save_path, best)
        logger.info("The model is saved to {}".format(patch_save_path))
        # tracker_save_pathが指定されている場合はpklで保存
        if tracker_save_path is not None:
            with open(tracker_save_path, "wb") as f:
                pickle.dump(fitness_tracker, f)
            logger.info(f"The tracker is saved to {tracker_save_path}")

        return best, fitness_tracker