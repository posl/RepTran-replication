# Experimental Plan for Adding PRoViTLP as a Baseline

## 1. Goal

This experiment adds PRoViTLP as an additional baseline to evaluate REPTRAN more comprehensively.

The current baselines are:

* Random
* ARACHNE
* ARACHNEW

These baselines are search-based repair methods that modify weights inside the FFNs of the Transformer encoder.

In contrast, PRoViTLP is a solver-based repair method. It modifies only the final classification layer of a Vision Transformer. Therefore, this experiment compares REPTRAN with a repair method that targets a different part of the same ViT model.

The main goal is to answer the following question:

> Is repairing the final classification layer sufficient for our fault benchmarks, or is it more effective to repair FFN weights inside the Transformer encoder?

## 2. Positioning of PRoViTLP

PRoViTLP should be added as a solver-based baseline.

REPTRAN modifies the linear layers inside the FFNs of the Transformer encoder.

PRoViTLP modifies the final classification layer after the Transformer encoder.

Therefore, both methods modify linear layers, but they target different components of the ViT.

```text
ViT
├── Patch embedding
├── Transformer encoder blocks
│   ├── Multi-head self-attention
│   └── FFN
│       ├── W_bef
│       └── W_aft
│       ↑
│       REPTRAN / ARACHNE / Random
└── Classification head
    └── Linear layer
        ↑
        PRoViTLP
```

This comparison helps clarify whether internal FFN repair is necessary for improving repair generalization.

## 3. Target Model

Use the same original ViT models as the current experiments.

* Dataset: CIFAR-100
* Dataset: Tiny-ImageNet
* Model: ViT base patch16 224
* Fine-tuned model: same as the original REPTRAN experiments
* Repair target: final classification layer only

The Transformer encoder parameters are fixed during PRoViTLP repair.

## 4. Fault Benchmarks

Use the same 18 fault benchmarks as the current experiments.

The benchmarks are defined by:

* 2 datasets
* 3 misclassification types
* 3 ranks

Misclassification types:

1. SRC-TGT
2. TGT-FP
3. TGT-FN

Ranks:

* Rank 1
* Rank 2
* Rank 3

Thus, the total number of benchmarks is:

```text
2 datasets × 3 misclassification types × 3 ranks = 18 benchmarks
```

## 5. Repair Set for PRoViTLP

For each benchmark, use the same target misclassified samples in the repair set.

Let:

```text
I_repair_mis
```

be the target misclassified samples used for repair.

For PRoViTLP, each sample is represented as:

```text
(x, y)
```

where:

* x is the input image
* y is the true label

The repair set for PRoViTLP is:

```text
S = I_repair_mis
```

PRoViTLP aims to make all samples in S correctly classified.

## 6. Label Set K

PRoViTLP only modifies the columns of the final classification layer that correspond to labels appearing in the repair set.

Let K be the set of true labels in the repair set:

```text
K = { y | (x, y) ∈ S }
```

The definition of K depends on the misclassification type.

### SRC-TGT

In SRC-TGT, all samples have the same true target label.

```text
K = { tgt }
```

### TGT-FN

In TGT-FN, all samples are true samples of the target class.

```text
K = { tgt }
```

### TGT-FP

In TGT-FP, the model wrongly predicts the target class for samples from other classes.

The true labels may contain multiple classes.

```text
K = { y | (x, y) ∈ I_repair_mis }
```

Therefore, TGT-FP may have a larger K than SRC-TGT and TGT-FN.

## 7. LP Formulation

Let the final classification layer be:

```text
logits = vW + b
```

where:

* v is the feature vector before the final classification layer
* W is the weight matrix of the final classification layer
* b is the bias vector
* logits is the output vector before softmax

PRoViTLP fixes v and modifies only part of W and b.

Specifically, it modifies:

```text
W[:, K]
b[K]
```

The remaining parameters are fixed.

For each repair sample (x, y), PRoViTLP adds constraints so that the logit of the true label y becomes larger than all other logits.

Conceptually, the constraint is:

```text
logit_y > logit_j  for all j ≠ y
```

The objective is to minimize the change from the original final layer parameters:

```text
minimize ||W[:, K] - W'[:, K]|| + ||b[K] - b'[K]||
```

The repaired model is obtained by replacing only W[:, K] and b[K] with the optimized values.

## 8. Implementation Steps

For each of the 18 benchmarks:

1. Load the original fine-tuned ViT model.

2. Collect the repair samples.

```text
S = I_repair_mis
```

3. Run the ViT encoder and extract the feature vector before the final classification layer for each repair sample.

```text
v = Encoder(x)
```

4. Construct the label set K from the true labels in S.

5. Build the LP problem.

6. Solve the LP problem.

7. Update only the final classification layer.

8. Evaluate the patched model on the test set.

9. Record RR, BR, and Ttot.

## 9. Evaluation Metrics

Use the same metrics as the current REPTRAN experiments.

### Repair Rate

Repair Rate measures how many target misclassifications in the test set are corrected.

```text
RR = |{(x, y) ∈ I_test_mis | M'(x) = y}| / |I_test_mis|
```

### Break Rate

Break Rate measures how many originally correct samples become misclassified after repair.

```text
BR = |{(x, y) ∈ I_test_cor | M'(x) ≠ y}| / |I_test_cor|
```

### Total Execution Time

For PRoViTLP, the total execution time is:

```text
Ttot = Tfeature + Tsolve + Tupdate
```

where:

* Tfeature is the time to extract features before the final layer
* Tsolve is the LP solving time
* Tupdate is the time to update the final layer

In practice, Tupdate is very small, so the main costs are feature extraction and LP solving.

## 10. RQ Integration

### RQ1: Effectiveness

Add PRoViTLP to the RQ1 comparison table.

Current methods:

* REPTRAN
* ARACHNE
* RandomR
* RandomA

New methods:

* REPTRAN
* ARACHNE
* RandomR
* RandomA
* PRoViTLP

Report:

* RR
* BR

Expected interpretation:

* If PRoViTLP has lower RR than REPTRAN, this supports the claim that repairing FFN weights improves generalization beyond the repair set.
* If PRoViTLP has lower BR than REPTRAN, this suggests that final-layer repair is more conservative.
* If PRoViTLP achieves high RR and low BR, it becomes a strong baseline and should be discussed as evidence that final-layer repair is effective for some fault types.

### RQ2: Efficiency

Add PRoViTLP to the RQ2 time comparison.

Report:

* Tfeature
* Tsolve
* Ttot

Expected interpretation:

* PRoViTLP may be faster than search-based methods when K is small.
* PRoViTLP may become slower for TGT-FP if K is large.
* REPTRAN may still be competitive because DE often terminates early.

### RQ3: Number of Selected Weights

Do not include PRoViTLP in RQ3.

RQ3 studies the effect of the number of selected FFN weights.

PRoViTLP does not select FFN weights and does not use Nw.

Therefore, including PRoViTLP in RQ3 would be conceptually inappropriate.

### RQ4: Balance Coefficient

Do not include PRoViTLP in RQ4.

RQ4 studies the effect of the balance coefficient α in the DE fitness function.

PRoViTLP does not use DE and does not have α.

Therefore, PRoViTLP should be excluded from RQ4.

## 11. Additional Analysis

### 11.1 Repair Set Efficacy

Since PRoViTLP is designed to guarantee correctness on the repair set, we should report repair set accuracy after repair.

For each method, report:

```text
Repair set efficacy = accuracy on I_repair_mis after repair
```

This is especially important because PRoViTLP is expected to achieve 100% efficacy on the repair set if the LP is feasible.

This additional metric helps distinguish:

* repair set fitting
* test set generalization

### 11.2 Result by Misclassification Type

Analyze PRoViTLP separately for each misclassification type.

Expected behavior:

* SRC-TGT: likely favorable because K is small
* TGT-FN: likely favorable because K is small
* TGT-FP: potentially harder because K can contain multiple true labels

This analysis can clarify when final-layer repair is sufficient.

### 11.3 Result by |K|

For PRoViTLP, record the size of K for each benchmark.

Report:

```text
|K|
Tsolve
RR
BR
```

This helps explain scalability and effectiveness.

A larger K means more LP variables and constraints.

## 12. Statistical Tests

For RQ1, perform the same statistical tests as the current paper.

Use the Wilcoxon signed-rank test and Cliff's delta.

Add comparisons such as:

* REPTRAN vs. PRoViTLP
* ARACHNE vs. PRoViTLP
* RandomR vs. PRoViTLP

Metrics:

* RR
* BR

For RQ2, also compare Ttot using the same statistical test.

## 13. Expected Outcomes

The expected outcomes are as follows.

### Case 1: REPTRAN has higher RR than PRoViTLP

This is the most favorable case for the current paper.

Possible interpretation:

```text
PRoViTLP guarantees correctness on the repair set, but its repair is restricted to the final classification layer. REPTRAN achieves better repair generalization by modifying FFN weights inside the Transformer encoder.
```

### Case 2: PRoViTLP has lower BR than REPTRAN

This is also expected.

Possible interpretation:

```text
PRoViTLP is more conservative because it modifies only the final classification layer and minimizes parameter changes. However, this conservativeness may limit its ability to generalize repairs to unseen test samples.
```

### Case 3: PRoViTLP outperforms REPTRAN

This result would require careful discussion.

Possible interpretation:

```text
For some fault benchmarks, final-layer decision boundary adjustment is sufficient. This suggests that not all misclassification types require internal FFN repair.
```

In this case, the paper should emphasize that REPTRAN is useful especially when internal representation repair is necessary.

## 14. Changes to the Paper

### Section II: Background and Related Work

Add a short explanation of PRoViT as a ViT-specific provable repair method.

Emphasize that PRoViT targets the final classification layer, while REPTRAN targets FFNs inside the Transformer encoder.

### Section IV-E: Baselines

Add PRoViTLP as a new baseline.

Suggested text:

```text
We also include PRoViTLP as a solver-based baseline for Vision Transformer repair. PRoViTLP modifies only the final classification layer of a Vision Transformer by solving a linear programming problem. Unlike REPTRAN and ARACHNE, which modify weights in the FFNs of the Transformer encoder, PRoViTLP does not modify the encoder blocks. This baseline allows us to examine whether final-layer repair is sufficient for our fault benchmarks.
```

### Section IV-F: Configuration

Add the configuration of PRoViTLP.

Suggested text:

```text
For PRoViTLP, we use the target misclassified samples in the repair set as the repair specification. We extract the feature vector before the final classification layer for each repair sample and construct an LP that modifies only the columns of the final-layer weight matrix and bias vector corresponding to the true labels appearing in the repair set. The objective minimizes the change from the original final-layer parameters.
```

### Section V-A: RQ1

Add PRoViTLP to the effectiveness table.

Discuss whether final-layer repair is sufficient for test repair generalization.

### Section V-B: RQ2

Add PRoViTLP to the execution time table.

Discuss the time cost of LP solving compared with DE-based search.

### Section VI: Discussion

Add a discussion comparing internal FFN repair and final-layer repair.

Suggested discussion point:

```text
The comparison with PRoViTLP highlights the difference between repairing the model's final decision boundary and repairing internal Transformer representations. When PRoViTLP achieves low break rates but limited repair generalization, it suggests that final-layer repair is conservative but insufficient for correcting the underlying internal behavior. In contrast, REPTRAN can achieve stronger repair generalization by modifying FFN weights, although this may increase the risk of side effects.
```

## 15. Summary

PRoViTLP should be added as a solver-based baseline in RQ1 and RQ2.

It should not be included in RQ3 or RQ4 because it does not select FFN weights and does not use DE.

The key comparison is:

```text
REPTRAN: internal FFN repair
PRoViTLP: final classification layer repair
```

This comparison can strengthen the paper by showing whether Transformer-specific FFN repair is necessary beyond final-layer adjustment.
