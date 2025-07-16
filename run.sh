#!/bin/bash

# # 멀티에이전트 실험 5번 병렬실행
# for i in {1..5}; do
#     python GSM8K/gsm_inference.py \
#         --model_1 qwen_A \
#         --model_2 qwen_B \
#         --model_3 qwen_C \
#         --output_dir GSM8K &
# done

# # 멀티에이전트 cat attack 실험 5번 병렬실행
# for i in {1..5}; do
#     python GSM8K/gsm_inference_cat_adversarial.py \
#         --model_1 qwen_A \
#         --model_2 qwen_B \
#         --model_3 qwen_C \
#         --output_dir GSM8K &
# done

# wait # 백그라운드에서 멀티에이전트 실행 다 끝나면

# 싱글에이전트 실험 5번 병렬실행
for i in {1..5}; do
    python GSM8K/gsm_inference_single.py \
        --model_1 qwen_A \
        --output_dir GSM8K &
done

# 싱글에이전트 cat attack 실험 5번 병렬실행
for i in {1..5}; do
    python GSM8K/gsm_inference_single_cat_adversarial.py \
        --model_1 qwen_A \
        --output_dir GSM8K &
done

wait # 백그라운드에서 싱글 에이전트 실행 다 끝나면
echo "모든 실험이 끝났습니다!"