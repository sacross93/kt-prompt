import pandas as pd
import openai
import os
from dotenv import load_dotenv
import time
import random
import re

# 환경 변수 로드
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

def load_system_prompt(file_path):
    """시스템 프롬프트 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def test_classification_gpt4o(system_prompt, test_sentences, model="gpt-4o", temperature=0.4):
    """GPT-4o를 사용한 문장 분류 테스트"""
    # 입력 형식 생성
    user_input = ""
    for i, sentence in enumerate(test_sentences, 1):
        user_input += f"{i}. {sentence}\n"
    
    try:
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input.strip()}
            ],
            temperature=temperature,
            max_tokens=1000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

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

def run_gpt4o_test(system_prompt_path, sample_size=20, temperature=0.4):
    """GPT-4o 테스트 실행"""
    # 데이터 로드
    df = pd.read_csv('data/samples.csv')
    
    # 랜덤 샘플 선택
    sample_df = df.sample(n=sample_size)
    
    # 시스템 프롬프트 로드
    system_prompt = load_system_prompt(system_prompt_path)
    
    print(f"=== GPT-4o 시스템 프롬프트 테스트 ({sample_size}개 샘플) ===")
    print(f"모델: gpt-4o")
    print(f"Temperature: {temperature}")
    print(f"프롬프트 길이: {len(system_prompt)}자")
    print(f"한글 비율: {calculate_korean_ratio(system_prompt):.1%}")
    print()
    
    # 테스트 실행
    test_sentences = sample_df['user_prompt'].tolist()
    ground_truth = sample_df['output'].tolist()
    
    print("테스트 실행 중...")
    start_time = time.time()
    result = test_classification_gpt4o(system_prompt, test_sentences, temperature=temperature)
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
            # "1. 사실형,긍정,현재,확실" 형식 처리
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
            
            # 대회 점수 계산
            korean_ratio = calculate_korean_ratio(system_prompt)
            length_score = min(1.0, (3000 - len(system_prompt)) / 3000 + 0.5)  # 길이 점수 추정
            total_score = 0.8 * accuracy + 0.1 * korean_ratio + 0.1 * length_score
            print(f"예상 대회 점수: {total_score:.1%}")
            
            return accuracy
        
        # 상세 비교 (처음 10개만)
        print("\n=== 상세 비교 (처음 10개) ===")
        for i in range(min(10, len(predictions), len(ground_truth))):
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
        return 0.0

def iterative_improvement(target_accuracy=0.8, max_iterations=5):
    """반복적으로 프롬프트를 개선하여 목표 정확도 달성"""
    
    prompts_to_test = [
        'system_prompt_optimized.txt',
        'system_prompt_v1.txt', 
        'system_prompt_v2.txt',
        'system_prompt_final.txt'
    ]
    
    best_accuracy = 0.0
    best_prompt = None
    
    for i, prompt_file in enumerate(prompts_to_test):
        print(f"\n{'='*60}")
        print(f"테스트 {i+1}/{len(prompts_to_test)}: {prompt_file}")
        print('='*60)
        
        try:
            accuracy = run_gpt4o_test(prompt_file, sample_size=30, temperature=0.4)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_prompt = prompt_file
                
            print(f"\n현재 최고 정확도: {best_accuracy:.1%} ({best_prompt})")
            
            if accuracy >= target_accuracy:
                print(f"\n🎉 목표 정확도 {target_accuracy:.1%} 달성!")
                print(f"최종 선택: {prompt_file} (정확도: {accuracy:.1%})")
                return prompt_file, accuracy
                
        except Exception as e:
            print(f"테스트 실패: {str(e)}")
            continue
    
    print(f"\n최종 결과: 최고 정확도 {best_accuracy:.1%} ({best_prompt})")
    return best_prompt, best_accuracy

if __name__ == "__main__":
    # 목표 정확도 0.8 달성까지 반복 테스트
    best_prompt, final_accuracy = iterative_improvement(target_accuracy=0.8)
    
    if final_accuracy >= 0.8:
        print(f"\n✅ 성공: {best_prompt}로 {final_accuracy:.1%} 정확도 달성!")
    else:
        print(f"\n⚠️  목표 미달성: 최고 {final_accuracy:.1%} 정확도")
        print("추가 프롬프트 개선이 필요합니다.")