#!/bin/bash

# 첫 번째 실험 5번 반복
for i in {1..5}; do
    python GSM8K/gsm_inference_single.py \
        --model_1 qwen_A \
        --output_dir GSM8K &
done

# 두 번째 실험 5번 반복
for i in {1..5}; do
    python GSM8K/gsm_inference_single_cat_adversarial.py \
        --model_1 qwen_A \
        --output_dir GSM8K &
done

# 모든 백그라운드 작업이 끝날 때까지 대기
wait
echo "모든 실험이 끝났습니다!"