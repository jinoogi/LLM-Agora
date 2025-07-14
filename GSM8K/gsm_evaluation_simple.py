import json
import numpy as np
import re

# ===== 설정 부분 =====
# 채점할 파일명을 여기에 직접 입력
RESULT_FILE = "GSM8K/gsm_result_qwen_qwen_qwen_20250713_071040.json"  # 여기에 파일명 입력

# 모델명들 (파일명에서 추출하거나 직접 입력)
MODEL_1 = "qwen"
MODEL_2 = "qwen" 
MODEL_3 = "qwen"

# COT 사용 여부 (파일명에 _cot이 있으면 True)
USE_COT = "_cot" in RESULT_FILE

# 출력 디렉토리
OUTPUT_DIR = "GSM8K"
# =====================

def solve_math_problems(input_str):
    pattern = r"\d+\.?\d*"
    matches = re.findall(pattern, input_str)
    if matches:
        return float(matches[-1])
    return None

def parse_answer(input_str):
    # \boxed{} 패턴을 찾음
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_match = re.search(boxed_pattern, input_str)
    if boxed_match:
        solution = boxed_match.group(1)
        # 숫자만 추출
        solution = re.sub(r"[^0-9.]", "", solution)
        if solution:
            return float(solution)
    return None

def answer_check(List, answer):
    # print("predictions:", List, "\tanswer:", answer)
    # 맞춘 개수를 리스트 길이로 나누어 정확도 계산
    correct_count = sum(1 for pred in List if pred == answer)
    accuracy = correct_count / len(List)
    return accuracy

def compute_accuracy(gt, pred_solutions):
    answers = solve_math_problems(gt)
    if not answers:
        return None
    
    if type(pred_solutions) == list:
        pred_answers = []
        for pred_solution in pred_solutions:
            pred_answer = parse_answer(pred_solution)
            if not pred_answer:
                pred_answer = solve_math_problems(pred_solution)
            pred_answers.append(pred_answer)
        return answer_check(pred_answers, answers)

def main():
    model_list = [MODEL_1, MODEL_2, MODEL_3]
    
    print(f"채점할 파일: {RESULT_FILE}")
    print(f"모델들: {model_list}")
    print(f"COT 사용: {USE_COT}")
    
    # 파일 읽기
    try:
        with open(RESULT_FILE, "r") as f:
            response_dict = json.load(f)
        print(f"총 {len(response_dict)}개 문제 로드됨")
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {RESULT_FILE}")
        return
    except Exception as e:
        print(f"❌ 파일 읽기 오류: {e}")
        return

    questions = [response_dict[i]["question"] for i in range(len(response_dict))]
    performance = []

    # 각 라운드별 성능 계산
    for turn in range(3):
        accuracies = []
        for idx in range(len(questions)):
            responses = [response_dict[idx]["agent_response"][model][turn] for model in model_list]
            gt = response_dict[idx]["answer"]

            accurate = compute_accuracy(gt, responses)
            if accurate is not None:
                accuracies.append(float(accurate))
            else:
                accuracies.append(0.0)

        performance.append({f"{turn+1}_performance": np.mean(accuracies)})
        print(f"라운드 {turn+1} 성능: {np.mean(accuracies):.4f}")

    # # 결과 저장
    # cot_suffix = "_cot" if USE_COT else ""
    # performance_file = f"{OUTPUT_DIR}/gsm_performance{cot_suffix}_simple.json"
    
    # try:
    #     with open(performance_file, "w") as f:
    #         json.dump(performance, f, indent=4)
    #     print(f"✅ 성능 결과 저장됨: {performance_file}")
    # except Exception as e:
    #     print(f"❌ 파일 저장 오류: {e}")

    # print("채점 완료!")

if __name__ == "__main__":
    main() 