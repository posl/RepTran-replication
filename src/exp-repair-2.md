# exp-repair-2: repair setを使ったLoRA

# 実験方法
誤分類の種類; 5種類．
- all : 全ての誤分類
- src_tgt : 正解ラベルsrcをtgtというラベルと間違える (srcとtgtは異なるラベル)
- tgt_all : ラベルtgtが関与する全ての間違い (any-to-tgt or tgt-to-any)
- tgt_fp : ラベルtgtに関するfp (誤検知, any-to-tgt)
- tgt_fn : ラベルtgtに関するfn (見逃し, tgt-to-any)

それぞれの種類でターゲットとなるrepair setだけを使ってLoRA as repairする

# 必要なスクリプト
- `exp-repair-2-1.py`: LoRA as repairを実行する. 
- `exp-repair-2-2.py`: いろんな設定でのイテレーションを作って， `exp-repair-2-1.py` のメイン関数を実行することを繰り返す. 
- `exp-repair-2-3.py`: 上のrepair結果をまとめる． 
- `exp-repair-2-4.py`: いろんな設定でのイテレーションを作って， `exp-repair-2-3.py` を繰り返す
- `exp-repair-2-5.py`: `exp-repair-2-4.py` の結果を表形式でまとめる．

# 評価方法
各誤分類の種類ごとに，RR (対象の誤分類のRR), BR (全体のBR) を出す？

# 実験結果

最終レイヤだけに固定する場合，いじる重みの数は `(3072 x r) x 2`