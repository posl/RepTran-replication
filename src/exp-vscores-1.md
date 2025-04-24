NLCの論文見て思った

正解と不正解サンプルに対してVscoreは本当に違うのか？

`exp-vscores-1-2.py`
- Vscoreやintermediate stateの保存をラベルごとに行う

`exp-vscores-1-3.ipynb`
- cor/misそれぞれで，ラベルごとのVscoreの類似度行列をヒートマップで表示
- t-SNEによるcor/misのVscoreの違いの可視化

`exp-vscores-1-4.py`
- Vscore計算時に正例と負例の数を合わせる

`exp-vscores-1-5.ipynb`
- `exp-vscores-1-4.py` を可視化
- 1-3から追加で，ラベルごとの正解/不正解時のVdiffのプロットもあります．

ここまででラベルごとの正解/不正解のVScoreには違いがあることは言える．
ただ間違い種類ごととなるとどうか？

`exp-vscores-1-6.ipynb`
- 間違いの種類 (src_tgt, tgt, tgt_fp, tgt_fn) ごとに Ineg と Ipos を特定． Inegはダメな振る舞い，Iposはお手本にしたい振る舞いに対応する．
- Vscore(Ineg)とVscore(Ipos)の差の絶対値としてVdiffを定義して，プロットする．

`exp-vscores-1-7.ipynb`
- 間違いの種類 (src_tgt, tgt, tgt_fp, tgt_fn) ごとに以下を行う．
    - Ineg と Ipos を特定． Inegはダメな振る舞い，Iposはお手本にしたい振る舞いに対応する．
    - Ineg と Ipos から，重みごとに Arachne の BL のスコアと，対応するニューロンのVDiffを取得．これで重みごとに3つの値が得られる．
    - BLに使われるfwd_imp, grad_loss, Vdiffの相関をプロット（3つの2次元散布図）