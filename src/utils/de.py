import pickle, os, sys, time, math, copy
# Add path to import utils
import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
from utils.data_util import format_label, make_batch_of_label

from logging import getLogger

logger = getLogger("base_logger")

def check_new_weights(patch, pos_before, pos_after, old_model, new_model, device=torch.device("cuda"), op=None):
    # The first len(pos_before) elements of patch are weights of pos_before in intermediate, and the subsequent len(pos_after) elements are weights of pos_after in output
    assert len(patch) == len(pos_before) + len(pos_after), "The length of patch should be equal to the sum of the lengths of pos_before and pos_after"
    num_pos_before = len(pos_before)
    num_pos_after = len(pos_after)
    # Iterate over pos_before and after respectively
    for ba, pos in enumerate([pos_before, pos_after]):
        list_for_comparison = []
        for model in [old_model, new_model]:
            # Set index of patch_candidate
            if ba == 0:
                idx_patch_candidate = range(0, num_pos_before)
                tgt_weight_data = model.base_model_last_layer.intermediate.dense.weight.data # This is a destructive change
            else:
                idx_patch_candidate = range(num_pos_before, num_pos_before + num_pos_after)
                tgt_weight_data = model.base_model_last_layer.output.dense.weight.data
            # Rewrite neurons at positions specified by pos
            xi, yi = pos[:, 0], pos[:, 1]
            list_for_comparison.append(tgt_weight_data[xi, yi])
        # Check if weights have changed
        sum_new_weights = list_for_comparison[1]
        sum_old_weights = list_for_comparison[0]
        if op is "sup":
            check_val = torch.sum(sum_new_weights).item() # Check if the sum of new weights is 0 since we're setting them to 0
        elif op is "enh":
            diff = list_for_comparison[1] - list_for_comparison[0] # new-old
            check_val = torch.sum(diff - list_for_comparison[0]).item() # Since we're doubling, new - 2 old should be 0
        # Check if check_val is 0
        assert np.isclose(check_val, 0, atol=1e-8), f"check_val should be close to 0 (check_val: {check_val})"

def set_new_weights(patch, pos_before, pos_after, model, device=torch.device("cuda"), op=None):
    # The first len(pos_before) elements of patch are weights of pos_before in intermediate, and the subsequent len(pos_after) elements are weights of pos_after in output
    assert len(patch) == len(pos_before) + len(pos_after), "The length of patch should be equal to the sum of the lengths of pos_before and pos_after"
    num_pos_before = len(pos_before)
    num_pos_after = len(pos_after)
    # Iterate over pos_before and after respectively
    for ba, pos in enumerate([pos_before, pos_after]):
        # In the method of taking pareto_front (BL), len(pos_before) or len(pos_after) may be 0
        if len(pos) == 0:
            continue
        # Set index for patch_candidate
        if ba == 0:
            idx_patch_candidate = range(0, num_pos_before)
            tgt_weight_data = model.base_model_last_layer.intermediate.dense.weight.data # This is a destructive change
        else:
            idx_patch_candidate = range(num_pos_before, num_pos_before + num_pos_after)
            tgt_weight_data = model.base_model_last_layer.output.dense.weight.data
        # Rewrite neurons at positions specified by pos
        xi, yi = pos[:, 0], pos[:, 1]
        if op is None:
            tgt_weight_data[xi, yi] = torch.from_numpy(patch[idx_patch_candidate]).to(device)
        elif op is "enh" or op is "enhance":
            tgt_weight_data[xi, yi] *= 2
        elif op is "sup" or op is "suppress":
            tgt_weight_data[xi, yi] *= 0
        elif isinstance(op, int):
            tgt_weight_data[xi, yi] *= op
        else:
            NotImplementedError(f"{op} is not supported yet")

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
        batch_labels,
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
        is_multi_label=True,
        pop_size=50,
        mode="neuron",
        weight_before2med=None,
        weight_med2after=None,
        pos_before=None,
        pos_after=None,
        custom_bounds=None
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
        self.is_multi_label = is_multi_label
        self.maximum_fitness = 0.0  # the maximum fitness value
        self.mdl = copy.deepcopy(partial_model) # Ensure that the original model doesn't change no matter how much the model is modified within DE_searcher
        self.max_search_num = max_search_num
        self.indices_to_sampled_correct = None
        self.tgt_pos = 0 # TODO: should not be hard coded
        self.pop_size = pop_size
        self.mode = mode
        self.custom_bounds = custom_bounds
        # Set labels
        self.batched_labels = batch_labels
        self.ground_truth_labels = np.concatenate(batch_labels, axis=0)

        # fitness computation related initialisation
        self.fitness = 0.0

        # deap related settings
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

        # Processing when mode is weight
        if mode == "weight":
            self.weight_before2med = weight_before2med
            self.weight_med2after = weight_med2after
            # Also display min and max
            logger.info(f"self.weight_before2med.shape: {self.weight_before2med.shape}, min: {np.min(self.weight_before2med)}, max: {np.max(self.weight_before2med)}")
            logger.info(f"self.weight_med2after.shape: {self.weight_med2after.shape}, min: {np.min(self.weight_med2after)}, max: {np.max(self.weight_med2after)}")
            # Calculate mean and standard deviation for DE initial value generation
            self.mean_b2m = np.mean(self.weight_before2med)
            self.std_b2m = np.std(self.weight_before2med)
            self.mean_m2a = np.mean(self.weight_med2after)
            self.std_m2a = np.std(self.weight_med2after)
            # Get length of pos_before, after
            self.num_pos_before = len(pos_before)
            self.num_pos_after = len(pos_after)
            self.num_total_pos = self.num_pos_before + self.num_pos_after
        
        # If mode is neither neuron nor weight, exit with error
        if mode not in ["neuron", "weight"]:
            NotImplementedError("mode should be either 'neuron' or 'weight'")

        logger.info("Finish Initializing DE_searcher...")

    def set_bounds(self, init_weight_values, custom_bounds="Arachne", v_orig=None):
        """
        Set the bounds for the search space.
        NOTE: The bounds set in this method are shared for one layer. We can set bounds for each weight by adopting different approach.
        """
        assert custom_bounds is not None, "custom_bounds should be set"
        if custom_bounds == "Arachne":
            min_v = np.min(init_weight_values)
            min_v = min_v * 2 if min_v < 0 else min_v / 2

            max_v = np.max(init_weight_values)
            max_v = max_v * 2 if max_v > 0 else max_v / 2

            bounds = (min_v, max_v)
            return bounds
        
        elif custom_bounds == "ContrRep":
            assert v_orig is not None, "v_orig should be set when custom_bounds is 'ContrRep'"
            
            min_v = v_orig * 2 if v_orig < 0 else v_orig / 2
            max_v = v_orig * 2 if v_orig > 0 else v_orig / 2

            bounds = (min_v, max_v)
            return bounds
        else:
            NotImplementedError(f"{custom_bounds} is not supported yet")

    def eval_neurons(self, patch_candidate, places_to_fix, show_log=True):
        # Predict data from self.inputs and return fitness function value using loss
        # When predicting data, change neurons at places_to_fix positions to patch_candidate values for prediction
        all_proba = []
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none") # NOTE: To output loss for each sample in the batch. Default reduction="mean" would give batch average
        for cached_state, y in zip(self.batch_hs_before_layernorm, self.batched_labels):
            logits = self.mdl(hidden_states_before_layernorm=cached_state, tgt_pos=self.tgt_pos, imp_pos=places_to_fix, imp_op=patch_candidate)
            # Convert output logits to probabilities
            proba = torch.nn.functional.softmax(logits, dim=-1)
            all_proba.append(proba.detach().cpu().numpy())
            # Calculate loss for each sample
            loss = loss_fn(proba, torch.from_numpy(y).to(self.device)).cpu().detach().numpy()
            losses_of_all.append(loss)
        losses_of_all = np.concatenate(losses_of_all, axis=0) # (num_of_data, )
        all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
        all_pred_laebls = np.argmax(all_proba, axis=-1) # (num_of_data, )

        # Evaluate whether prediction results are correct
        is_correct = all_pred_laebls == self.ground_truth_labels
        # Get number of samples that were originally correct and remained correct
        num_intact = sum(is_correct[self.indices_to_correct])
        # Get number of samples that were originally incorrect and became correct
        num_patched = sum(is_correct[self.indices_to_wrong])
        # Get average loss for originally correct samples
        losses_of_correct = np.mean(losses_of_all[self.indices_to_correct])
        # Get average loss for originally incorrect samples
        losses_of_wrong = np.mean(losses_of_all[self.indices_to_wrong])

        fitness_for_correct = (num_intact / len(self.indices_to_correct) + 1) / (losses_of_correct + 1)
        fitness_for_wrong = (num_patched / len(self.indices_to_wrong) + 1) / (losses_of_wrong + 1)
        final_fitness = self.alpha * fitness_for_correct + fitness_for_wrong

        if show_log:
            logger.info(f"num_intact: {num_intact}/{len(self.indices_to_correct)}, num_patched: {num_patched}/{len(self.indices_to_wrong)}")
        return (final_fitness,)

    def eval_weights(self, patch_candidate, pos_before, pos_after, show_log=True):
        assert len(patch_candidate) == self.num_total_pos, "The length of patch_candidate should be equal to the number of total positions"
        set_new_weights(patch=patch_candidate, pos_before=pos_before, pos_after=pos_after, model=self.mdl, device=self.device)
        # Predict data from self.inputs and return fitness function value using loss
        all_proba = []
        losses_of_all = []
        loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
        for cached_state, y in zip(self.batch_hs_before_layernorm, self.batched_labels):
            logits = self.mdl(hidden_states_before_layernorm=cached_state, tgt_pos=self.tgt_pos)
            # Convert output logits to probabilities
            proba = torch.nn.functional.softmax(logits, dim=-1)
            all_proba.append(proba.detach().cpu().numpy())
            # Calculate loss for each sample
            loss = loss_fn(proba, torch.from_numpy(y).to(self.device)).cpu().detach().numpy()
            losses_of_all.append(loss)
        losses_of_all = np.concatenate(losses_of_all, axis=0) # (num_of_data, )
        losses_to_correct = losses_of_all[self.indices_to_correct]
        losses_to_wrong = losses_of_all[self.indices_to_wrong]
        all_proba = np.concatenate(all_proba, axis=0) # (num_of_data, num_of_classes)
        all_pred_labels = np.argmax(all_proba, axis=-1) # (num_of_data, )
        # Evaluate whether prediction results are correct
        is_correct = all_pred_labels == self.ground_truth_labels # Data indices with correct predictions in the modified model
        indices_to_correct_to_correct = is_correct[self.indices_to_correct] # Indices of samples that were originally correct and remained correct
        indices_to_correct_to_wrong = ~indices_to_correct_to_correct # Indices of samples that were originally correct and became incorrect
        indices_to_wrong_to_correct = is_correct[self.indices_to_wrong] # Indices of samples that were originally incorrect and became correct
        indices_to_wrong_to_wrong = ~indices_to_wrong_to_correct # Indices of samples that were originally incorrect and remained incorrect
        assert sum(indices_to_correct_to_correct) + sum(indices_to_correct_to_wrong) == len(self.indices_to_correct), f"The sum of indices_to_correct_to_correct and indices_to_correct_to_wrong should be equal to the length of indices_to_correct (sum(indices_to_correct_to_correct): {sum(indices_to_correct_to_correct)}, sum(indices_to_correct_to_wrong): {sum(indices_to_correct_to_wrong)}, len(self.indices_to_correct): {len(self.indices_to_correct)}"
        assert sum(indices_to_wrong_to_correct) + sum(indices_to_wrong_to_wrong) == len(self.indices_to_wrong), f"The sum of indices_to_wrong_to_correct and indices_to_wrong_to_wrong should be equal to the length of indices_to_wrong (sum(indices_to_wrong_to_correct): {sum(indices_to_wrong_to_correct)}, sum(indices_to_wrong_to_wrong): {sum(indices_to_wrong_to_wrong)}, len(self.indices_to_wrong): {len(self.indices_to_wrong)}"

        # TODO: The form of fitness_fn has subtle variations, so we want to make it customizable.
        # Get number of samples that were originally correct and remained correct
        num_intact = sum(indices_to_correct_to_correct)
        # Get number of samples that were originally incorrect and became correct
        num_patched = sum(indices_to_wrong_to_correct)
        # Get average loss for originally correct samples
        mean_of_losses_of_correct = np.mean(losses_of_all[self.indices_to_correct])
        # Get average loss for originally incorrect samples
        mean_of_losses_of_wrong = np.mean(losses_of_all[self.indices_to_wrong])
        # Loss for samples that were originally correct but became incorrect
        losses_of_correct_to_wrong = losses_to_correct[indices_to_correct_to_wrong]
        # Loss for samples that were originally incorrect and remained incorrect
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
        # Both fitness_for_correct and fitness_for_wrong are in the range [0, 1]
        assert 0 <= fitness_for_correct <= 1, f"fitness_for_correct should be in [0, 1] (fitness_for_correct: {fitness_for_correct})"
        assert 0 <= fitness_for_wrong <= 1, f"fitness_for_wrong should be in [0, 1] (fitness_for_wrong: {fitness_for_wrong})"
        # print(f"num_intact: {num_intact}/{len(self.indices_to_correct)}, num_patched: {num_patched}/{len(self.indices_to_wrong)}")
        # print(f"fitness_for_correct: {fitness_for_correct}, fitness_for_wrong: {fitness_for_wrong}")
        final_fitness = (1-self.alpha) * fitness_for_correct + self.alpha * fitness_for_wrong
        # Boldly use only intact_rate and patched_rate
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

        # Data for observing DE progress
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

        # Initial value generation for neuron modification (since it's x times the neuron, sample initial values from N(1, 1))
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
            bounds = []
            # self.mean_valuesとself.std_valuesはそれぞれ，最初のself.num_pos_before個についてはself.mean_b2m, その後のself.num_pos_after個についてはself.mean_m2a のようにする (標準偏差も同様)
            for i in range(self.num_total_pos):
                if i < self.num_pos_before:
                    self.mean_values.append(self.mean_b2m)
                    self.std_values.append(self.std_b2m)
                    if self.custom_bounds is None:
                        bounds.append((None, None))
                    else:
                        bounds.append(self.set_bounds(self.weight_before2med, custom_bounds=self.custom_bounds, v_orig=self.weight_before2med[pos_before[i][0], pos_before[i][1]]))
                else:
                    self.mean_values.append(self.mean_m2a)
                    self.std_values.append(self.std_m2a)
                    if self.custom_bounds is None:
                        bounds.append((None, None))
                    else:
                        bounds.append(self.set_bounds(self.weight_med2after, custom_bounds=self.custom_bounds, v_orig=self.weight_med2after[pos_after[i-self.num_pos_before][0], pos_after[i-self.num_pos_before][1]]))
            assert len(bounds) == self.num_total_pos, f"len(bounds): {len(bounds)}, self.num_total_pos: {self.num_total_pos}"
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
        for key, value in best.tracking_dict.items():
                fitness_tracker[key].append(value)

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
                logger.info(f"Processing {new_model_name}...")

                # select
                target_indices = [_i for _i in range(pop_size) if _i != pop_idx]
                a_idx, b_idx, c_idx = toolbox.select(target_indices)
                a = pop[a_idx]
                b = pop[b_idx]
                c = pop[c_idx]

                y = toolbox.clone(ind) # 値渡し

                index = self.random.randrange(num_places_to_fix)
                for i, value in enumerate(ind):
                    # 一部を変異させる
                    if i == index or self.random.random() < self.recombination:
                        # パッチ候補の値は対象レイヤの重みから計算したバウンドに収める
                        y[i] = np.clip(a[i] + MU * (b[i] - c[i]), bounds[i][0], bounds[i][1])
                        # y[i] = a[i] + MU * (b[i] - c[i])

                # eval_ret = toolbox.evaluate(ind, *args_for_eval) # (*) BUG: ind だと変異させた y で評価してなくない? 
                eval_ret = toolbox.evaluate(y, *args_for_eval) # NOTE: こっちが正しい?
                y.fitness.values = (eval_ret[0], )
                y.tracking_dict = eval_ret[1]
                logger.info(f"[{new_model_name}] fitness: {y.fitness.values[0]}")
                if y.fitness.values[0] >= ind.fitness.values[0]:  # better
                    # (*) BUG: (*) だと絶対ここに入る (上の判定が==なので).
                    logger.info(f"[{new_model_name}] y.fitness.values[0] ({y.fitness.values[0]}) >= ind.fitness.values[0] ({ind.fitness.values[0]}) is TRUE")
                    pop[pop_idx] = y  # upddate # (*) BUG: (*) だと絶対ここで個体が更新されてしまう
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
            # fitness_trackerのキーに対応する値をログ表示
            for key in fitness_tracker.keys():
                    logger.info(f"{key}: {fitness_tracker[key]}")
            # この時点でのbestで予測した結果が, fitness_tracker["term1_pos"][-1]やfitness_tracker["term1_neg"]との一貫性確認
            logger.info("Evaluating the best model at this generation...")
            _, tmp_tracker_dict = eval_func(best, *args_for_eval, show_log=True)
            tmp_num_intact, tmp_num_patched = tmp_tracker_dict["term1_pos"], tmp_tracker_dict["term1_neg"]
            logger.info(f"tmp_num_intact, tmp_num_patched: {tmp_num_intact}, {tmp_num_patched}")
            assert tmp_num_intact == best.tracking_dict["term1_pos"], f"tmp_num_intact: {tmp_num_intact}, best.tracking_dict['term1_pos']: {best.tracking_dict['term1_pos']}"
            assert tmp_num_patched == best.tracking_dict["term1_neg"], f"tmp_num_patched: {tmp_num_patched}, best.tracking_dict['term1_neg']: {best.tracking_dict['term1_neg']}"

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

        # bestをnpyでSave
        np.save(patch_save_path, best)
        logger.info("The model is saved to {}".format(patch_save_path))
        # tracker_save_pathが指定されている場合はpklでSave
        if tracker_save_path is not None:
            with open(tracker_save_path, "wb") as f:
                pickle.dump(fitness_tracker, f)
            logger.info(f"The tracker is saved to {tracker_save_path}")

        return best, fitness_tracker