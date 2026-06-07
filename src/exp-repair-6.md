これに関するrepairの実験は全て `exp-repair-6-{実験サブID}` という形式でレポジトリに対応させる．
スクリプトのファイル名は `exp-repair-6-{実験サブID}-{スクリプトID}.py` とする．

> [!NOTE] このページでタスク管理風なこともしたい．
> 🥚 : 未着手
> 🏃 : 実行中
> 🏝️ : 実装は終わって実行待ち
> ✅ : 完了

# RQ5でやること：ハイパラpの影響
## モチベーション
RQ1,2ではp=0.5（ModFIとModGLを等重みで組み合わせる）だったが，それを変えた場合の影響を調査するのがこのRQ．
査読コメントへの対応として，p=0.5という設定が恣意的でないことを実験的に示す．

## 設定
- pのバリエーション: {0.1, 0.5, 0.9}
  - 0.1はModGL重視，0.9はModFI重視，0.5はRQ1のデフォルト
- N_w = 236（RQ4/exp-repair-5と同じ，RQ3でgood balanceとなった値）
- α = 10（RQ1のデフォルト，10/11に変換して使用）

## RQ4（exp-repair-5）との違い
- αはsearchフェーズのみに影響するためlocalizationを再実行不要だった
- **pはselectionフェーズに影響するため，localizationから再実行が必要**
- そのため，localizationとrepairを1スクリプトにまとめる

# `exp-repair-6-1`
## 目的
上のpの設定でlocalizationとrepairを実行する．

## ステップ
### ステップ1：Localization + Repairの実行
- exp-repair-4-1-1.py（localization）とexp-repair-5-1-1.py（repair）の処理をまとめて実行する
- pをパラメータとして受け取り，`calculate_top_n_flattened`に `weight_grad_loss=1-p, weight_fwd_imp=p` として渡す
- α=10（実装上は10/11），N_w=236 で固定
- 各pで5回繰り返し（randomnessのため）
- 該当スクリプト: `exp-repair-6-1-1.py`（実行），`exp-repair-6-1-2.py`（ランチャー）

### ステップ2：テストセットで評価してデータ記録
- exp-repair-5-1-3.py, -4.py と同様の評価スクリプト
- 該当スクリプト: `exp-repair-6-1-3.py`，`exp-repair-6-1-4.py`

### ステップ3：記録したデータを図表にまとめて可視化
- exp-repair-5-1-5.py と同様にRR, BRの折れ線グラフを作成
- x軸: p値 {0.1, 0.5, 0.9}，y軸: RR / BR
- 該当スクリプト: `exp-repair-6-1-5.py`

# 進捗表
| ステップ | サブタスク                    | スクリプト名                         | C100 | tiny-imagenet |
| ---- | ------------------------ | ------------------------------ | ---- | ------------- |
| 1    | Localization + Repairの実行 | `exp-repair-6-1-1.py`, `-2.py` | 🥚   | 🥚            |
| 2    | テストセットで評価・記録             | `exp-repair-6-1-3.py`, `-4.py` | 🥚   | 🥚            |
| 3    | 結果の図表化と可視化               | `exp-repair-6-1-5.py`          | 🥚   | 🥚            |
