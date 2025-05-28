#!/usr/bin/env bash
# ------------------------------------------------------------------
#  sync_metrics.sh
#    LoRA 実験と比較するために，リモート EC2 マシンから
#    exp-repair-1-metrics_for_*.json だけを再帰コピーする
#
#    misclf_top{1,2,3}/
#        └─ {src_tgt,tgt_fp,tgt_fn,tgt}_repair_weight_by_de/**/exp-repair-1-metrics_for_*.json
#
#  使い方:
#    chmod +x sync_metrics.sh
#    ./sync_metrics.sh
# ------------------------------------------------------------------

set -euo pipefail

REMOTE_HOST="restore-ec2-nnrepair"
REMOTE_USER="ubuntu"                         # 変更する場合ここ
REMOTE_BASE="/home/ubuntu/ishimoto-transformer-analysis/src/out_vit_c100_fold0"
LOCAL_BASE="/home/ishimoto/ishimoto-transformer-analysis/src/out_vit_c100_fold0"

# SSH 鍵が必要なら SSH_OPTS='-e "ssh -i ~/.ssh/id_ed25519"'
SSH_OPTS=""

# misclf_top と misclassification type のリスト
TOP_IDS=(1 2 3)
MISCLF_TYPES=(src_tgt tgt_fp tgt_fn tgt)

for top in "${TOP_IDS[@]}"; do
  for mt in "${MISCLF_TYPES[@]}"; do
    REMOTE_DIR="${REMOTE_BASE}/misclf_top${top}/${mt}_repair_weight_by_de/"
    LOCAL_DIR="${LOCAL_BASE}/misclf_top${top}/${mt}_repair_weight_by_de/"

    # 必要ならローカル側のディレクトリを作成
    mkdir -p "${LOCAL_DIR}"

    echo "==> syncing ${REMOTE_DIR} → ${LOCAL_DIR}"
    rsync -avz --progress ${SSH_OPTS} \
      --prune-empty-dirs \
      --include='*/' \
      --include='exp-repair-1-metrics_for_*json' \
      --exclude='*' \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}" \
      "${LOCAL_DIR}"
  done
done

echo "✓  全て完了しました"
