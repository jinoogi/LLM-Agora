import json
import numpy as np
import re
import os
import argparse
from glob import glob

# ========== 유틸 함수 ==========
def find_all_json_files(folder):
    '''
    하위 폴더까지 모든 json 파일 탐색
    '''
    #  os.path.join(folder, '*.json)'로 LLM-Agora/GSM8K/score_test/*.json 처럼 만들어주고, glob으로 실제 파일경로들의 리스트 반환
    return glob(os.path.join(folder,'*.json'))

def extract_model_names(agent_response_dict):
    # agent_response의 key가 모델명
    return list(agent_response_dict.keys())

def extract_last_boxed(text):
    # 모든 \boxed{...}를 찾아서 마지막 것만 반환
    # ()는 추출대상, []는 그룹이라 }이 아닌게 1개 이상인걸 추출함
    matches = re.findall(r'\\boxed\{([^}]+)\}', text)
    if matches:
        return matches[-1].strip()
    return None

def is_correct(pred, gt):
    # 정답 비교 (숫자만 추출해서 비교)
    try:
        pred_num = float(re.sub(r'[^0-9.\-]', '', pred))
        gt_num = float(re.sub(r'[^0-9.\-]', '', gt))
        return pred_num == gt_num
    except:
        return False

def score_json_file(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    if not data:
        return None
    # 첫 문제의 agent_response에서 모델명 추출
    model_names = extract_model_names(data[0]['agent_response'])
    # 라운드 수 추출 (모델별 응답 리스트 길이)
    num_rounds = len(next(iter(data[0]['agent_response'].values())))
    # 각 라운드별, 모델별 정답률
    results = {model: [0.0]*num_rounds for model in model_names}
    total = len(data)
    for model in model_names:
        for round_idx in range(num_rounds):
            correct = 0
            for ex in data:
                pred = ex['agent_response'][model][round_idx]
                last_boxed = extract_last_boxed(pred)
                # "... #### 15" 형태로 answer 텍스트에서 실제 정답 추출 
                gt = re.search(r'####\s*([0-9\.\-]+)', ex["answer"])
                if gt != None and last_boxed and is_correct(last_boxed, gt.group(1)):
                    correct += 1
            results[model][round_idx] = round(float(correct) / total, 3)
    return {
        'file': os.path.basename(json_path),
        # 'models': model_names,
        # 'num_rounds': num_rounds,
        'accuracy': results,
        # 'total': total
    }

def main():
    json_files = find_all_json_files("/root/LLM-Agora/GSM8K/score_test")
    print(f"총 {len(json_files)}개 json 파일 발견!")
    all_results = []
    for path in json_files:
        print(f"채점 중: {path}")
        try:
            res = score_json_file(path)
            if res:
                all_results.append(res)
        except Exception as e:
            print(f"[오류] {path}: {e}")
    # with open("/root/LLM-Agora/GSM8K/score_output", 'w') as f:
    #     json.dump(all_results, f, indent=2, ensure_ascii=False)

        # 1. 먼저 JSON을 들여쓰기가 적용된 문자열로 변환합니다.
    output_string = json.dumps(all_results, indent=2, ensure_ascii=False)

    # 2. 정규식을 사용해 여러 줄로 나뉜 숫자 리스트를 한 줄로 합칩니다.
    #    ("qwen_A" 와 같은 키) : [ ... ] 형태의 패턴을 명시적으로 찾아 수정합니다.
    compact_list_string = re.sub(
        r'("qwen_[ABC]"): \[\s*(.*?)\s*\]',  # "qwen_A/B/C" 키를 명시적으로 찾습니다.
        lambda m: f"{m.group(1)}: [{', '.join(''.join(m.group(2).split()).split(','))}]",
        output_string,
        flags=re.DOTALL
    )
    
    # 3. 최종적으로 수정된 문자열을 파일에 씁니다.
    with open("/root/LLM-Agora/GSM8K/score_output", 'w', encoding='utf-8') as f:
        f.write(compact_list_string)
    print("모든 채점 결과가 /root/LLM-Agora/GSM8K/score_output에 저장되었습니다!")

if __name__ == "__main__":
    main() 

