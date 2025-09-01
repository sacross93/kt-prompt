"""
GPT-4o 최종 검증기
Gemini 2.5 Flash에서 0.7점 이상 달성한 프롬프트를 GPT-4o로 최종 검증
"""

import os
import json
import pandas as pd
import openai
from typing import List, Dict, Tuple
from dataclasses import dataclass
import time
import re
from dotenv import load_dotenv

load_dotenv()

@dataclass
class ValidationResult:
    """검증 결과"""
    model_name: str
    prompt_name: str
    accuracy: float
    total_questions: int
    correct_answers: int
    attribute_accuracies: Dict[str, float]
    improvement_vs_baseline: float
    test_duration: float

class GPT4oValidator:
    """GPT-4o를 이용한 최종 성능 검증"""
    
    def __init__(self, data_path: str = "data/samples.csv"):
        self.data_path = data_path
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        # OpenAI 클라이언트 설정
        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)
        
        # 데이터 로드
        self.samples = pd.read_csv(data_path)
        print(f"샘플 데이터 로드: {len(self.samples)}개")
        
        # 기존 최고 성능 (비교 기준)
        self.baseline_performance = 0.7  # system_prompt_final.txt 성능
    
    def validate_with_gpt4o(self, prompt_path: str, sample_size: int = 50) -> ValidationResult:
        """GPT-4o로 최종 검증"""
        print(f"\n=== GPT-4o 최종 검증: {os.path.basename(prompt_path)} ===")
        start_time = time.time()
        
        # 프롬프트 로드
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # 테스트 샘플 선택
        test_samples = self.samples.head(sample_size)
        sentences = test_samples['user_prompt'].tolist()
        answers = test_samples['output'].tolist()
        
        # GPT-4o로 배치 처리
        all_predictions = []
        batch_size = 10
        
        for i in range(0, len(sentences), batch_size):
            batch_sentences = sentences[i:i+batch_size]
            
            # 입력 텍스트 생성
            user_content = ""
            for j, sentence in enumerate(batch_sentences, 1):
                user_content += f"{j}. {sentence}\n"
            
            try:
                # GPT-4o API 호출
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,
                    max_tokens=1000
                )
                
                response_text = response.choices[0].message.content
                
                # 결과 파싱
                batch_predictions = self._parse_results(response_text, len(batch_sentences))
                all_predictions.extend(batch_predictions)
                
                print(f"배치 {i//batch_size + 1} 완료 ({len(batch_sentences)}개)")
                
                # API 호출 간격
                time.sleep(1)
                
            except Exception as e:
                print(f"GPT-4o API 호출 오류: {e}")
                # 오류 시 빈 결과 추가
                all_predictions.extend([""] * len(batch_sentences))
        
        # 정확도 계산
        accuracy, attribute_accuracies = self._calculate_accuracy(all_predictions, answers)
        
        # 기존 성능 대비 개선도 계산
        improvement = accuracy - self.baseline_performance
        
        test_duration = time.time() - start_time
        
        result = ValidationResult(
            model_name="GPT-4o",
            prompt_name=os.path.basename(prompt_path),
            accuracy=accuracy,
            total_questions=len(sentences),
            correct_answers=int(accuracy * len(sentences)),
            attribute_accuracies=attribute_accuracies,
            improvement_vs_baseline=improvement,
            test_duration=test_duration
        )
        
        print(f"GPT-4o 검증 완료:")
        print(f"- 정확도: {accuracy:.3f} ({result.correct_answers}/{result.total_questions})")
        print(f"- 기존 대비 개선: {improvement:+.3f}")
        print(f"- 속성별 정확도: {attribute_accuracies}")
        print(f"- 소요 시간: {test_duration:.1f}초")
        
        return result
    
    def _parse_results(self, response: str, num_questions: int) -> List[str]:
        """결과 파싱"""
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # "번호. 유형,극성,시제,확실성" 형식 파싱
            match = re.match(r'(\d+)\.\s*([^,]+),([^,]+),([^,]+),([^,\s]+)', line)
            if match:
                _, type_val, polarity, tense, certainty = match.groups()
                result = f"{type_val.strip()},{polarity.strip()},{tense.strip()},{certainty.strip()}"
                results.append(result)
        
        # 결과 개수 맞추기
        while len(results) < num_questions:
            results.append("")
        
        return results[:num_questions]
    
    def _calculate_accuracy(self, predictions: List[str], answers: List[str]) -> Tuple[float, Dict[str, float]]:
        """정확도 계산"""
        total = len(answers)
        correct = 0
        
        # 속성별 정확도 계산
        type_correct = polarity_correct = tense_correct = certainty_correct = 0
        
        for pred, ans in zip(predictions, answers):
            if pred == ans:
                correct += 1
            
            # 속성별 비교
            if pred and ans:
                pred_parts = pred.split(',')
                ans_parts = ans.split(',')
                
                if len(pred_parts) == 4 and len(ans_parts) == 4:
                    if pred_parts[0] == ans_parts[0]:
                        type_correct += 1
                    if pred_parts[1] == ans_parts[1]:
                        polarity_correct += 1
                    if pred_parts[2] == ans_parts[2]:
                        tense_correct += 1
                    if pred_parts[3] == ans_parts[3]:
                        certainty_correct += 1
        
        overall_accuracy = correct / total if total > 0 else 0
        attribute_accuracies = {
            "type": type_correct / total if total > 0 else 0,
            "polarity": polarity_correct / total if total > 0 else 0,
            "tense": tense_correct / total if total > 0 else 0,
            "certainty": certainty_correct / total if total > 0 else 0
        }
        
        return overall_accuracy, attribute_accuracies
    
    def compare_with_baseline(self, result: ValidationResult) -> Dict[str, any]:
        """기존 최고 성능과 비교"""
        comparison = {
            "current_performance": result.accuracy,
            "baseline_performance": self.baseline_performance,
            "improvement": result.improvement_vs_baseline,
            "improvement_percentage": (result.improvement_vs_baseline / self.baseline_performance) * 100,
            "is_improved": result.improvement_vs_baseline > 0,
            "significance": "significant" if abs(result.improvement_vs_baseline) >= 0.05 else "marginal"
        }
        
        return comparison
    
    def save_validation_result(self, result: ValidationResult, comparison: Dict[str, any]):
        """검증 결과 저장"""
        validation_data = {
            "model": result.model_name,
            "prompt": result.prompt_name,
            "accuracy": result.accuracy,
            "total_questions": result.total_questions,
            "correct_answers": result.correct_answers,
            "attribute_accuracies": result.attribute_accuracies,
            "baseline_comparison": comparison,
            "test_duration": result.test_duration,
            "timestamp": "2024-01-01"
        }
        
        with open("prompt/analysis/gpt4o_validation.json", 'w', encoding='utf-8') as f:
            json.dump(validation_data, f, ensure_ascii=False, indent=2)
        
        print("GPT-4o 검증 결과가 저장되었습니다.")
    
    def generate_final_report(self, result: ValidationResult, comparison: Dict[str, any]) -> str:
        """최종 리포트 생성"""
        
        status = "🎉 성능 개선 성공!" if comparison["is_improved"] else "📊 성능 유지"
        significance = comparison["significance"]
        
        report = f"""
# GPT-4o 최종 검증 리포트

## 검증 결과
- **모델**: {result.model_name}
- **프롬프트**: {result.prompt_name}
- **전체 정확도**: {result.accuracy:.3f} ({result.correct_answers}/{result.total_questions})

## 속성별 성능
- **유형 분류**: {result.attribute_accuracies['type']:.3f}
- **극성 분류**: {result.attribute_accuracies['polarity']:.3f}
- **시제 분류**: {result.attribute_accuracies['tense']:.3f}
- **확실성 분류**: {result.attribute_accuracies['certainty']:.3f}

## 기존 성능 대비 비교
- **기존 최고 성능**: {comparison['baseline_performance']:.3f}
- **현재 성능**: {comparison['current_performance']:.3f}
- **개선도**: {comparison['improvement']:+.3f} ({comparison['improvement_percentage']:+.1f}%)
- **개선 유의성**: {significance}

## 결론
{status}

{'성능이 유의미하게 개선되었습니다.' if comparison['is_improved'] and significance == 'significant' else '추가 최적화를 통해 더 나은 성능을 달성할 수 있습니다.'}

## 권장사항
{'현재 프롬프트를 최종 버전으로 채택하고 실제 운영에 적용할 수 있습니다.' if comparison['is_improved'] else '추가적인 프롬프트 엔지니어링을 통해 성능 개선을 시도해보세요.'}
"""
        
        return report

if __name__ == "__main__":
    try:
        validator = GPT4oValidator()
        
        # 최고 성능 프롬프트 검증
        best_prompt_path = "prompt/gemini/enhanced_v3.txt"
        
        if os.path.exists(best_prompt_path):
            print("=== GPT-4o 최종 검증 시작 ===")
            result = validator.validate_with_gpt4o(best_prompt_path, sample_size=30)
            
            # 기존 성능과 비교
            comparison = validator.compare_with_baseline(result)
            
            # 결과 저장
            validator.save_validation_result(result, comparison)
            
            # 최종 리포트 생성
            final_report = validator.generate_final_report(result, comparison)
            
            with open("prompt/analysis/final_validation_report.md", 'w', encoding='utf-8') as f:
                f.write(final_report)
            
            print("\n" + final_report)
            
        else:
            print(f"프롬프트 파일을 찾을 수 없습니다: {best_prompt_path}")
    
    except Exception as e:
        print(f"GPT-4o 검증 오류: {e}")
        print("OPENAI_API_KEY 환경 변수를 확인해주세요.")