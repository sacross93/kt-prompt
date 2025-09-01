import pandas as pd
import google.generativeai as genai
import os
from dotenv import load_dotenv
import time
import random
import re

# 환경 변수 로드
load_dotenv()
gemini_api_key = os.getenv('GEMINI_API_KEY')

def load_system_prompt(file_path):
    """시스템 프롬프트 파일 로드"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def test_classification_gemini(system_prompt, test_sentences, model="gemini-2.5-flash-preview-05-20", temperature=0.4):
    """Gemini를 사용한 문장 분류 테스트"""
    # API 키 설정
    genai.configure(api_key=gemini_api_key)
    
    # 입력 형식 생성
    user_input = ""
    for i, sentence in enumerate(test_sentences, 1):
        user_input += f"{i}. {sentence}\n"
    
    try:
        # 모델 초기화
        model_instance = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": 1000,
            }
        )
        
        # 시스템 프롬프트와 사용자 입력 결합
        full_prompt = f"{system_prompt}\n\n{user_input.strip()}"
        
        response = model_instance.generate_content(full_prompt)
        
        return response.text.strip()
        
    except Exception as e:
        return f"Error: {str(e)}"

def calculate_accuracy(predictions, ground_truth):
    """분류 정확도 계산"""
    if len(predictions) != len(ground_truth):
        print(f"길이 불일치: 예측 {len(predictions)}, 정답 {len(ground_truth)}")
        return 0.0, [0, 0, 0, 0]
    
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
    
    return overall_accuracy, attr_accuracies

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

def calculate_competition_score(accuracy, korean_ratio, length):
    """대회 점수 계산 (추정)"""
    # 길이 점수 추정 (935자 기준으로 최적화)
    if 800 <= length <= 1100:
        length_score = 1.0
    elif length < 800:
        length_score = length / 800
    else:
        length_score = 1100 / length
    
    total_score = 0.8 * accuracy + 0.1 * korean_ratio + 0.1 * length_score
    return total_score

def run_gemini_test(system_prompt_path, sample_size=30, temperature=0.4):
    """Gemini 테스트 실행"""
    # 데이터 로드
    df = pd.read_csv('data/samples.csv')
    
    # 랜덤 샘플 선택
    sample_df = df.sample(n=sample_size, random_state=42)
    
    # 시스템 프롬프트 로드
    system_prompt = load_system_prompt(system_prompt_path)
    
    print(f"=== Gemini 2.5 Flash 시스템 프롬프트 테스트 ({sample_size}개 샘플) ===")
    print(f"모델: gemini-2.5-flash-preview-05-20")
    print(f"Temperature: {temperature}")
    print(f"프롬프트 길이: {len(system_prompt)}자")
    print(f"한글 비율: {calculate_korean_ratio(system_prompt):.1%}")
    print()
    
    # 테스트 실행
    test_sentences = sample_df['user_prompt'].tolist()
    ground_truth = sample_df['output'].tolist()
    
    print("테스트 실행 중...")
    start_time = time.time()
    result = test_classification_gemini(system_prompt, test_sentences, temperature=temperature)
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
            accuracy, attr_accuracies = calculate_accuracy(predictions, ground_truth)
            print(f"전체 분류 정확도: {accuracy:.1%}")
            
            # 대회 점수 계산
            korean_ratio = calculate_korean_ratio(system_prompt)
            competition_score = calculate_competition_score(accuracy, korean_ratio, len(system_prompt))
            print(f"예상 대회 점수: {competition_score:.4f}")
            
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
            
            return accuracy, attr_accuracies, competition_score
            
    except Exception as e:
        print(f"결과 파싱 오류: {str(e)}")
        return 0.0, [0, 0, 0, 0], 0.0

def test_multiple_prompts(prompt_files, sample_size=30):
    """여러 프롬프트를 Gemini로 테스트하여 최적 프롬프트 찾기"""
    results = {}
    
    for prompt_file in prompt_files:
        print(f"\n{'='*60}")
        print(f"테스트: {prompt_file}")
        print('='*60)
        
        try:
            accuracy, attr_accuracies, competition_score = run_gemini_test(prompt_file, sample_size=sample_size, temperature=0.4)
            results[prompt_file] = {
                'accuracy': accuracy,
                'competition_score': competition_score,
                'type_acc': attr_accuracies[0],
                'polarity_acc': attr_accuracies[1],
                'tense_acc': attr_accuracies[2],
                'certainty_acc': attr_accuracies[3]
            }
        except Exception as e:
            print(f"테스트 실패: {str(e)}")
            continue
    
    # 결과 비교
    if len(results) > 1:
        print(f"\n{'='*80}")
        print("=== 프롬프트 성능 비교 (Gemini 2.5 Flash 기준) ===")
        print(f"{'프롬프트':<30} {'전체정확도':<12} {'대회점수':<12} {'극성정확도':<12} {'시제정확도':<12}")
        print("-" * 90)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['competition_score'], reverse=True)
        for prompt_file, result in sorted_results:
            name = prompt_file.replace('system_prompt_', '').replace('.txt', '')
            print(f"{name:<30} {result['accuracy']:<12.1%} {result['competition_score']:<12.4f} "
                  f"{result['polarity_acc']:<12.1%} {result['tense_acc']:<12.1%}")
        
        best_prompt = sorted_results[0][0]
        best_score = sorted_results[0][1]['competition_score']
        print(f"\n🏆 최고 성능: {best_prompt} (점수: {best_score:.4f})")
        print(f"💡 이 프롬프트를 GPT-4o로 최종 제출하세요!")
    
    return results

if __name__ == "__main__":
    # 여러 프롬프트 테스트로 최적 프롬프트 찾기
    prompt_files = [
        'system_prompt_v1.txt',
        'system_prompt_v1_enhanced.txt',
        'system_prompt_v2.txt',
        'system_prompt_optimized.txt',
        'system_prompt_best.txt'
    ]
    
    print("🔮 Gemini 2.5 Flash로 프롬프트 최적화 테스트")
    print("💡 목표: GPT-4o 제출용 최고 프롬프트 찾기")
    
    results = test_multiple_prompts(prompt_files, sample_size=30)