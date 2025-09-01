import pandas as pd
import requests
import json
import time
import random
import re

def load_system_prompt(file_path):
    """시스템 프롬프트 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def test_classification_local(system_prompt, test_sentences, base_url="http://192.168.120.102:11434", model="qwen3:32b"):
    """로컬 LLM을 사용한 문장 분류 테스트"""
    # 입력 형식 생성
    user_input = ""
    for i, sentence in enumerate(test_sentences, 1):
        user_input += f"{i}. {sentence}\n"
    
    # 전체 프롬프트 구성
    full_prompt = f"{system_prompt}\n\n{user_input.strip()}"
    
    # Ollama API 호출
    url = f"{base_url}/api/generate"
    
    payload = {
        "model": model,
        "prompt": full_prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 0.1
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        response_text = result.get('response', 'No response')
        
        # <think> </think> 블록 제거
        cleaned_response = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
        return cleaned_response.strip()
        
    except requests.exceptions.RequestException as e:
        return f"Error: {str(e)}"
    except json.JSONDecodeError as e:
        return f"JSON Error: {str(e)}"

def calculate_accuracy(predictions, ground_truth):
    """분류 정확도 계산"""
    if len(predictions) != len(ground_truth):
        print(f"길이 불일치: 예측 {len(predictions)}, 정답 {len(ground_truth)}")
        return 0.0
    
    total_correct = 0
    total_attributes = 0
    attribute_correct = [0, 0, 0, 0]  # 유형, 극성, 시제, 확실성
    
    for pred, truth in zip(predictions, ground_truth):
        pred_parts = pred.split(',')
        truth_parts = truth.split(',')
        
        if len(pred_parts) == 4 and len(truth_parts) == 4:
            for i, (p, t) in enumerate(zip(pred_parts, truth_parts)):
                if p.strip() == t.strip():
                    total_correct += 1
                    attribute_correct[i] += 1
                total_attributes += 1
    
    overall_accuracy = total_correct / total_attributes if total_attributes > 0 else 0.0
    
    # 속성별 정확도
    attr_names = ['유형', '극성', '시제', '확실성']
    attr_accuracies = []
    for i, correct in enumerate(attribute_correct):
        acc = correct / len(predictions) if len(predictions) > 0 else 0.0
        attr_accuracies.append(acc)
        print(f"{attr_names[i]} 정확도: {acc:.1%}")
    
    return overall_accuracy

def calculate_korean_ratio(text):
    """한글 문자 비율 계산"""
    korean_chars = 0
    total_chars = 0
    
    for char in text:
        if char not in [' ', '\n', '\t']:
            total_chars += 1
            if '가' <= char <= '힣':
                korean_chars += 1
    
    return korean_chars / total_chars if total_chars > 0 else 0.0

def run_sample_test(system_prompt_path, sample_size=10, model="qwen3:32b"):
    """샘플 테스트 실행"""
    # 데이터 로드
    df = pd.read_csv('data/samples.csv')
    
    # 랜덤 샘플 선택
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # 시스템 프롬프트 로드
    system_prompt = load_system_prompt(system_prompt_path)
    
    print(f"=== 로컬 LLM 시스템 프롬프트 테스트 ({sample_size}개 샘플) ===")
    print(f"모델: {model}")
    print(f"프롬프트 길이: {len(system_prompt)}자")
    print(f"한글 비율: {calculate_korean_ratio(system_prompt):.1%}")
    print()
    
    # 테스트 실행
    test_sentences = sample_df['user_prompt'].tolist()
    ground_truth = sample_df['output'].tolist()
    
    print("테스트 실행 중...")
    start_time = time.time()
    result = test_classification_local(system_prompt, test_sentences, model=model)
    end_time = time.time()
    
    print(f"응답 시간: {end_time - start_time:.1f}초")
    print()
    
    print("=== 모델 출력 ===")
    print(result)
    print()
    
    # 결과 파싱 및 정확도 계산
    try:
        predictions = []
        for line in result.split('\n'):
            line = line.strip()
            # "1.추론형,긍정,현재,확실" 또는 "1. 추론형,긍정,현재,확실" 형식 처리
            if line and re.match(r'^\d+\.', line):
                # 번호와 점 다음의 내용 추출
                parts = re.split(r'^\d+\.?\s*', line, 1)
                if len(parts) == 2:
                    pred = parts[1].strip()
                    predictions.append(pred)
        
        print(f"=== 성능 평가 ===")
        print(f"예측 개수: {len(predictions)}")
        print(f"실제 개수: {len(ground_truth)}")
        
        if len(predictions) > 0:
            accuracy = calculate_accuracy(predictions, ground_truth)
            print(f"전체 분류 정확도: {accuracy:.1%}")
        
        # 상세 비교
        print("\n=== 상세 비교 ===")
        for i in range(min(len(predictions), len(ground_truth))):
            print(f"{i+1}. {test_sentences[i]}")
            print(f"   예측: {predictions[i] if i < len(predictions) else 'N/A'}")
            print(f"   정답: {ground_truth[i]}")
            
            # 일치 여부 확인
            if i < len(predictions):
                pred_parts = predictions[i].split(',')
                truth_parts = ground_truth[i].split(',')
                if len(pred_parts) == 4 and len(truth_parts) == 4:
                    matches = [p.strip() == t.strip() for p, t in zip(pred_parts, truth_parts)]
                    match_str = " ".join(["✓" if m else "✗" for m in matches])
                    print(f"   일치: {match_str} (유형 극성 시제 확실성)")
            print()
            
    except Exception as e:
        print(f"결과 파싱 오류: {str(e)}")

def test_different_models():
    """다양한 모델로 테스트"""
    models = ["qwen3:32b", "qwen2.5:14b", "llama3.1:8b"]
    
    for model in models:
        print(f"\n{'='*50}")
        print(f"모델: {model}")
        print('='*50)
        try:
            run_sample_test('system_prompt_v1.txt', sample_size=5, model=model)
        except Exception as e:
            print(f"모델 {model} 테스트 실패: {str(e)}")

if __name__ == "__main__":
    # 기본 테스트
    run_sample_test('system_prompt_v1.txt', sample_size=10, model="qwen3:32b")