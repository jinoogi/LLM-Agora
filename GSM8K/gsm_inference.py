import requests
import json
import random
import time
from tqdm import tqdm
import argparse
import os
from datetime import datetime


def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_1", type=str)
    parser.add_argument("--model_2", type=str)
    parser.add_argument("--model_3", type=str)
    parser.add_argument("--round", default=2, type=int)
    parser.add_argument(
        "--cot",
        default=False,
        action='store_true',
        help="If this is True, you can use Chain-of-Thought during inference."
    )
    parser.add_argument(
        "--output_dir",
        default="Math",
        type=str,
        help="Directory to save the result file"
    )

    return parser.parse_args()

def load_json(prompt_path, endpoint_path):
    with open(prompt_path, "r") as prompt_file:
        prompt_dict = json.load(prompt_file)

    with open(endpoint_path, "r") as endpoint_file:
        endpoint_dict = json.load(endpoint_file)

    return prompt_dict, endpoint_dict

def summarize_message(agent_contexts, instruction, idx):
    prefix_string = "Here are a list of opinions from different agents: "

    for agent in agent_contexts:
        agent_response = agent[-1]["content"]
        response = "\n\n One agent response: ```{}```".format(agent_response)

        prefix_string = prefix_string + response

    return prefix_string

def generate_gsm(agents, question):
    agent_contexts = [[{"model": agent, "content": f"  Solve: {question} Answer: \\boxed{{answer}}. Do not say anything after providing boxed{{answer}}."}] for agent in agents]
    return agent_contexts

def read_jsonl(path: str):
    with open(path, "r") as fh:
        return [json.loads(line) for line in fh.readlines() if line]

if __name__ == "__main__":
    args = args_parse()
    model_list = [args.model_1, args.model_2, args.model_3]

    prompt_dict, endpoint_dict = load_json("src/prompt_template.json", "src/inference_endpoint.json")

    def generate_answer(model, formatted_prompt):
        API_URL = endpoint_dict[model]["API_URL"]
        headers = endpoint_dict[model]["headers"]
        payload = {
            "prompt": formatted_prompt,
            "max_tokens": 512,
            "temperature": 0.25,
            "repetition_penalty": 1.1,
            "enable_thinking": False
        }
        try:
            resp = requests.post(API_URL, json=payload, headers=headers)
            response = resp.json()
        except:
            print("retrying due to an error......")
            time.sleep(5)
            return generate_answer(model, formatted_prompt)
        
        return {"model": model, "content": response["choices"][0]["text"]}
    
    def prompt_formatting(model, instruction, cot):
        if model == "alpaca" or model == "orca":
            prompt = prompt_dict[model]["prompt_no_input"]
        else:
            prompt = prompt_dict[model]["prompt"]
        
        if cot:
            instruction += "Let's think step by step."

        return {"model": model, "content": prompt.format(instruction=instruction)}

    agents = len(model_list)
    rounds = args.round
    random.seed(0)

    evaluation = 30

    generated_description = []

    questions = read_jsonl("data/GSM8K/gsm8k_test.jsonl")
    random.shuffle(questions)

    for idx in tqdm(range(evaluation)):
        question = questions[idx]["question"]
        answer = questions[idx]["answer"]

        agent_contexts = generate_gsm(model_list, question)
        # print("원시구조:",agent_contexts)

        print(f"# Question No.{idx+1} starts...")

        message = []

        models_response_values = [[],[],[]]

        # Debate
        for debate in range(rounds+1):
            # Refer to the summarized previous response
            if debate == 0:
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], agent_contexts[i][0]["content"], args.cot))
            if debate != 0:
                message = []
                message.append(summarize_message(agent_contexts, question, 2 * debate - 1))
                # print("\n\t 메시지:",message)
                for i in range(len(agent_contexts)):
                    agent_contexts[i].append(prompt_formatting(agent_contexts[i][-1]["model"], agent_contexts[i][0]["content"]+message[0], args.cot))

            for i,agent_context in enumerate(agent_contexts):
                # Generate new response based on summarized response
                # print("\n\tagent_context:",agent_context[-1]["content"])
                completion = generate_answer(agent_context[-1]["model"], agent_context[-1]["content"])
                # print("\tcompletion:",completion)
                agent_context.append(completion)

                #모델 답변내역 추가
                # print("이거추가:",completion["content"],"\n끝")
                models_response_values[i].append(completion["content"])



        print(f"# Question No.{idx+1} debate is ended.")

        models_response = {
            f"{args.model_1}": models_response_values[0],
            f"{args.model_2}": models_response_values[1],
            f"{args.model_3}": models_response_values[2]
        }

        response_summarization = [
            message[0]
        ]
        generated_description.append({"question_id": idx, "question": question, "agent_response": models_response, "summarization": response_summarization, "answer": answer})

    # 자동 파일명 생성
    def generate_filename():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        models_str = "_".join([m.replace("/", "_") for m in model_list if m])
        cot_suffix = "_cot" if args.cot else ""
        base_name = f"gsm_result_temp0.25{cot_suffix}_{models_str}_{timestamp}.json"
        return base_name
    
    # 출력 디렉토리 생성
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 파일명 생성 및 충돌 방지
    filename = generate_filename()
    filepath = os.path.join(args.output_dir, filename)
    
    # 파일이 이미 존재하면 번호 추가
    counter = 1
    while os.path.exists(filepath):
        name, ext = os.path.splitext(filename)
        filename = f"{name}_v{counter}{ext}"
        filepath = os.path.join(args.output_dir, filename)
        counter += 1
    
    print(f"Saving result to: {filepath}")
    with open(filepath, "w") as f:
        json.dump(generated_description, f, indent=4)

    print("All done!!")