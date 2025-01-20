# exp-fl-5: 訓練途中のモデルを利用してFLベンチマークを作成した上での評価

Paoloとの打ち合わせを受けて発生．
exp-fl-4とは別のFLの実験．
FL for DNNは評価のベンチマークがないので間接的になりがち．そこを解決するために，学習途中のエポックでできたモデルにFLを行う．
そして，FLで特定された重みと，実際に学習で更新された重みがどれくらい近いかを見ることでFLの評価とする．
FLが学習にalignしている感じがするが，real faultsを特定するという感じがする．

## 実験方法
今回は2epoch学習している．なので，1epoch目の学習結果のモデルに対してFLを行うことになる．

# 必要なスクリプト
- `exp-fl-5-1.py`: M1からM2に進化する時に変更された重みを確認 (各重み値に対して変更前後の差の絶対値を取ってランキング)．これにより，FLのベンチマークを作成できる．
    - 将来変更されるべきをFLで予測できるか，というのがポイント．
- `exp-fl-5-2.py`: M1に対する誤分類の種類とか色々を取得する (misclf_info)
    - 結果は `exp-fl-5/c100_fold{k}` 以下に保存される．
- `exp-fl-5-3.py`: 各種法の適用のために，各サンプルに対するFFN直前のLayerNorm前の状態とFFNの中間状態をキャッシュとして保存する．実行時間などは `src/out_vit_c100_fold0/logs/exp-fl-5-3.log` から確認できる．
- M1に全てのFL手法を適用する
    - neuron-level: random (`exp-fl-5-7.py`), ig (`exp-fl-5-4.py`), vdiff (`exp-fl-5-6.py`), vdiff+mean_act (`exp-fl-5-6.py`)
    - weight-level: random (`exp-fl-5-7.py`), bl (`exp-fl-5-5.py`), vdiff (`exp-fl-5-6.py`), vdiff+mean_act+grad_loss (`exp-fl-5-6.py`)
        - nを指定した場合はそれがファイル名からわかるようにしたい．nを指定しない場合はnの情報をファイル名から落とすことで，そのこともわかるようにしたい．
    - ここで忘れずに各手法の時間を測りたい．`exp-fl-X-Y_time.csv` の形式で保存しておく．
- vdiff+mean_act+grad_loss (`exp-fl-5-6.py`)について：
    - Vdiffとmean_actによるニューロン特定 -> grad_lossも使った重み特定をやる
    - 保存したいもの：cor, mis時のvscore, mean_act (mis時のみ), 各重みに対するgrad_lossのnpy


## 実験結果