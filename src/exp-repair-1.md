# exp-repair-1: ViT for C100のリペア

# 実験方法
誤分類の種類; 5種類．
- all : 全ての誤分類
- src_tgt : 正解ラベルsrcをtgtというラベルと間違える (srcとtgtは異なるラベル)
- tgt_all : ラベルtgtが関与する全ての間違い (any-to-tgt or tgt-to-any)
- tgt_fp : ラベルtgtに関するfp (誤検知, any-to-tgt)
- tgt_fn : ラベルtgtに関するfn (見逃し, tgt-to-any)

それぞれの種類で特定した重みに対するDEを適用する．

# 必要なスクリプト
- `exp-repair-1-1.py`: DEアルゴリズムによるリペアを実行する. 007eが元々．
- `exp-repair-1-2.py`: いろんな設定でのイテレーションを作って， `exp-repair-1-1.py` のメイン関数を実行することを繰り返す. 007fが元々.
- `exp-repair-1-3.py`: 上のrepair結果をまとめる． 007gを参考にする．

# 評価方法
各誤分類の種類ごとに，RR (対象の誤分類のRR), BR (全体のBR) を出す？

# 実験結果

