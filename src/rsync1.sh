for i in 1 2 3 4 5; do
  rsync -av \
    --prune-empty-dirs \
    --include='*/' \
    --include='*.npy' \
    --exclude='*' \
    restore-ec2-nnrepair:/home/ubuntu/ishimoto-transformer-analysis/src/out_vit_c100_fold0/misclf_top${i}/vscores \
    /home/ishimoto/ishimoto-transformer-analysis/src/out_vit_c100_fold0/misclf_top${i}
done
