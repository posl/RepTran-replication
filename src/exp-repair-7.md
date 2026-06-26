これに関するrepairの実験は全て `exp-repair-7-{実験サブID}` という形式でレポジトリに対応させる．
スクリプトのファイル名は `exp-repair-7-{実験サブID}-{スクリプトID}.py` とする．
これはISSRE 2026 major revision の **Meta3（neuron-scoreの寄与をequal-budget ablationで切り分ける）** への対応実験．

> [!NOTE] このページでタスク管理風なこともしたい．
> 🥚 : 未着手
> 🏃 : 実行中
> 🏝️ : 実装は終わって実行待ち
> ✅ : 完了

# RQ6でやること：neuron scoreの構成要素のablation（equal-budget）
## モチベーション
査読コメント Meta3 への対応．
我々のFLは，weight-levelのArachneスコア（grad_loss / fwd_imp）に対して，
**neuron score** を掛けて重みを選ぶ．このneuron scoreは2成分の要素積で構成される（`localize_neurons_with_mean_activation` 参照）：

```
neuron_score = VDiff × MisAct
  VDiff  = |vscore_cor - vscore_mis| を min-max正規化したもの（正解/誤分類のv-score差）
  MisAct = 対象誤分類サンプルにわたる中間ニューロン活性化の平均を min-max正規化したもの
```

査読者の懸念は「neuron scoreの寄与が本当にあるのか」「あるとしてどちらの成分が効くのか」．
これを **重み予算を完全に揃えた（equal-budget）状態** で切り分けるのが本RQ．

> [!WARNING] リスク
> RQ3のTable VIIで，Nw=472では Ours（=Full）と ARACHNEW（=neuron score無し）のRR差が
> 統計的に有意でないという結果が既に出ている．本RQはその点を正面から扱う実験なので，
> 「成分を分けても差が出ない可能性」を念頭に，結果がどう転んでも response letter で
> 説明できる枠組み（効果量＋ベンチマーク別の内訳）で記述する．

## 設定（4条件，全て equal-budget = Nw=472 を選択）
| 条件 | neuron score | 既存 / 新規 | 由来 |
| --- | --- | --- | --- |
| **Full** | `VDiff × MisAct` | 既存（再利用） | Ours（`exp-repair-4-1_location_n472_weight_ours.npy`） |
| **VDiff-only** | `VDiff`（MisActを一様1に） | **新規** | 本RQで実行 |
| **MisAct-only** | `MisAct`（VDiffを一様1に） | **新規** | 本RQで実行 |
| **No-neuron-score** | 一様1（neuron score不使用） | 既存（再利用） | ARACHNEW/ArachneV3（`..._n472_weight_bl.npy`） |

- N_w = **472**（RQ2/exp-repair-4 の 0.01%，good balance値）に固定．4条件すべてこの予算で重みを選ぶ → 予算が交絡しない．
- α = 10（実装上 10/11），bounds=Arachne，p は weight-level の話なので本RQでは default（=0.5相当の組合せ）を踏襲．neuron-levelのablationと直交．
- 各条件 5回繰り返し（DEのrandomness）．
- 誤分類ベンチマーク 9種 × 5 reps × 2新条件 × 2データセット = **180 run**（既存2条件は再利用するので追加実行不要）．

## exp-repair-6 との違い
- exp-repair-6（p感度）は **weight-levelの選択フェーズ** を変えるのでlocalizationから再実行が必要だった．
- 本RQは **neuron scoreの構成** を変える＝同じくlocalizationから再実行が必要．
- よって exp-repair-6 と同じく **localization + repair を1スクリプトにまとめた runner/launcher 構成** を踏襲する．

# `exp-repair-7-1`
## 目的
VDiff-only / MisAct-only の2条件で，Nw=472・localization＋repairを実行する．

## ステップ
### ステップ0：localize関数にscore_modeを追加 ✅
- `utils/vit_util.py` の `localize_neurons_with_mean_activation` に
  `score_mode ∈ {"full","vdiff","misact"}` 引数を追加する（default `"full"` で既存挙動を変えない）．
  - `"full"`   : `neuron_score = vmap_diff_abs * mean_activation`（現行）
  - `"vdiff"`  : `neuron_score = vmap_diff_abs`（mean_activationを一様1扱い）
  - `"misact"` : `neuron_score = mean_activation`（vmap_diff_absを一様1扱い）
  - min-max正規化は各成分について現行どおり行い，使う成分だけを返す．
- 既存呼び出し（exp-repair-4-1-1.py 等）は引数省略時に従来と同一の結果になることを確認．

### ステップ1：Localization + Repairの実行 🏝️
- `exp-repair-6-1-1.py`（runner）をベースに，`--p` の代わりに `--score_mode {vdiff,misact}` を受け取るよう改造．
  - localizationで `localize_neurons_with_mean_activation(..., score_mode=args.score_mode)` を呼ぶ．
  - Nw=472, α=10/11, bounds=Arachne 固定．
  - location保存名・patch保存名に `score_mode` を含める（例 `exp-repair-7-1-location_n472_{score_mode}_weight_ours.npy`）．
- 該当スクリプト: `exp-repair-7-1-1.py`（runner），`exp-repair-7-1-2.py`（launcher）
  - launcherは exp-repair-6-1-2.py 同様，subprocess・retry・per-benchmark incremental save+resume・`USE_TF=0` を踏襲．

### ステップ2：テストセットで評価してデータ記録 🏝️
- exp-repair-6-1-3.py / -4.py と同様の評価スクリプト．`score_mode` を設定キーに含める．
- 該当スクリプト: `exp-repair-7-1-3.py`，`exp-repair-7-1-4.py`

### ステップ3：4条件を統合して図表化・統計検定 🏝️
- 既存の Full（ours, n472）と No-neuron-score（bl, n472）の test結果を読み込み，
  新規2条件と結合して4条件のRR/BRを1つのデータフレームに集約．
- 可視化：ベンチマーク別＋全体で RR・BR を4条件で並べる（棒グラフ / box）．
- 統計検定：ベンチマークをペアにして
  **Wilcoxon signed-rank + Cliff's delta** を
  - Full vs VDiff-only
  - Full vs MisAct-only
  - Full vs No-neuron-score（RQ3の再掲・整合確認）
  - VDiff-only vs MisAct-only
  の4対比で実施し，RR・BRそれぞれ p値と効果量を1テーブルに．
- 該当スクリプト: `exp-repair-7-1-5.py`
- 出力名（exp-repair-6-1の命名に倣う）:
  - `exp-repair-7-1_{c100,tiny-imagenet}_test_results_all.csv`（4条件×9×5の生データ）
  - `exp-repair-7-1_{c100,tiny-imagenet}_test_stats.csv`（4対比×{RR,BR}のp値＋Cliff's delta）
  - `exp-repair-7-1_{c100,tiny-imagenet}_ablation_plots.pdf`

## 期待される議論（response letter向け）
- Full が両単独成分を有意に上回るなら → 2成分の相補性を主張（neuron scoreの設計が効く）．
- 単独成分のどちらかが Full と同等なら → その成分が主たる寄与で，もう一方は補助と記述．
- 4条件とも有意差なし（RQ3と整合）なら → 「限られたNw予算では neuron score の*構成*が
  RR/BRを大きく動かさない＝頑健であり，無闇な複雑化をしていない」ことを正直に述べ，
  neuron scoreの価値はFL全体（RQ1/RQ3でのArachne比改善）で示済みと位置づける．
  いずれの結末でも equal-budget で切り分けた事実が Meta3 への直接回答になる．

# 進捗表
| ステップ | サブタスク | スクリプト名 | C100 | tiny-imagenet |
| ---- | ---- | ---- | ---- | ---- |
| 0 | localize関数に score_mode 追加 | `utils/vit_util.py` | ✅ | ✅ |
| 1 | Localization + Repair（VDiff/MisAct） | `exp-repair-7-1-1.py`, `-2.py` | 🏝️ | 🏝️ |
| 2 | テストセットで評価・記録 | `exp-repair-7-1-3.py`, `-4.py` | 🏝️ | 🏝️ |
| 3 | 4条件統合・図表化・統計検定 | `exp-repair-7-1-5.py` | 🏝️ | 🏝️ |
