import json
import re
import numpy as np
import csv
from collections import defaultdict

with open('GSM8K/score_output', 'r') as f:
    results = json.load(f)

grouped = defaultdict(list) # grouped['a'] 이런식으로 없는 키에 접근해도 자동으로 값에 빈 리스트 []를 만들어준다고 함

for res in results:
    fname = res['file']
    acc = res['accuracy']

    # 실험 조건 추출
    is_single = 'single' in fname
    is_cat = 'cat_adversarial' in fname
    temp_match = re.search(r'temp([0-9\.]+)', fname)
    temp = float(temp_match.group(1)) if temp_match else None # group(0)는 매칭된 전체문자열이고, group(1)을 해야 첫 ()안의 내용임

    # multi agent면 qwen_A/B/C 평균, single이면 qwen_A만
    if is_single:
        vals = acc['qwen_A']
    else:
        vals = np.mean([acc['qwen_A'], acc['qwen_B'], acc['qwen_C']], axis=0)
    grouped[(is_single, is_cat, temp)].append(vals) 
    '''
    grouped는 이런식으로 생겼을거임
    {
        (True, False, 0.25): [
            [0.5, 0.6, 0.7],
            [0.53, 0.62, 0.68],
            [0.51, 0.61, 0.69]
        ],
        (False, True, 0.5): [
            [0.4, 0.5, 0.6],
            [0.42, 0.52, 0.62]
        ]
        # ... (다른 조건들도 같은 방식)
    }
    '''
# CSV 파일로 저장
with open('GSM8K/score.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # writer.writerow(['agent_type', 'cat_attack', 'temp', 'round', 'mean', 'std'])
    # for cond, vals_list in grouped.items():
    #     arr = np.array(vals_list)  # shape: (5, 라운드수)
    #     mean = np.mean(arr, axis=0)
    #     std = np.std(arr, axis=0)
    #     agent_type = 'single' if cond[0] else 'multi'
    #     cat_attack = cond[1]
    #     temp = cond[2]
    #     for round_idx, (m, s) in enumerate(zip(mean, std)):
    #         writer.writerow([agent_type, cat_attack, temp, round_idx, round(m, 3), round(s, 3)])
    writer.writerow(['agent_type', 'cat_attack', 'temp', 'round', 'score'])
    for cond, batch in grouped.items():
        agent_type = 'single' if cond[0] else 'multi'
        cat_attack = cond[1]
        temp = cond[2]
        for round_scores in batch:
            for round_idx,score in enumerate(round_scores):
                writer.writerow([agent_type, cat_attack, temp, round_idx, round(score,3)])
print("GSM8K/summary.csv 파일이 생성되었습니다!")