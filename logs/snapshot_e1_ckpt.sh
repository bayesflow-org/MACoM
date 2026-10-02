#!/usr/bin/env bash
# Keep a copy of every e1_train checkpoint save (the trainer overwrites one file).
cd /home/jerry/Documents/research/projects/CogFormer-2
CK=outputs/checkpoints/model_class/cf/r4c4_b128_e15000/cogformer_model_class_fm_l8_h8_p256_s32_d128_b128_e15000_t100.pt
D=outputs/checkpoints/model_class/cf/r4c4_b128_e15000_snapshots
last=$(stat -c %Y "$CK")
while true; do
  sleep 60
  [[ -e "$CK.done" ]] && { cp "$CK" "$D/epoch15000_final.pt"; echo "$(date '+%F %T') final saved; exiting"; exit 0; }
  now=$(stat -c %Y "$CK" 2>/dev/null || echo "$last")
  if [[ "$now" != "$last" ]]; then
    sleep 120   # let torch.save finish writing
    ep=$(.venv/bin/python -c "import torch,sys; print(torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$CK" 2>/dev/null | tail -n 1)
    [[ "$ep" =~ ^[0-9]+$ ]] && cp "$CK" "$D/epoch$(printf %05d "$ep").pt" && echo "$(date '+%F %T') saved epoch $ep"
    last=$(stat -c %Y "$CK")
  fi
done
