"""
최적화된 Gemini 테스터
API 할당량을 고려한 효율적인 테스트 시스템
"""

import os
import json
import pandas as pd
import google.generativeai as genai
from typing import List, Dict, Tuple
import time
import re
from dotenv import load_dotenv

load_dotenv()

class OptimizedGeminiTester:
    """API 할당량을 고려한 최적화된 테스터"""
    
    def __init__(self, data_path: str = "data/samples.csv"):
        self.data_path = data_path
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        self.samples = pd.read_csv(data_path)
        print(f"샘플 데이터 로드: {len(self.samples)}개")
    
    def test_single_prompt_efficiently(self, prompt_path: str, sample_size: int = 20) -> Dict:
        """단일 프롬프트를 효율적으로 테스트"""
        print(f"\n=== {os.path.basename(prompt_path)} 효율적 테스트 ===")
        
        with open(prompt_path, 'r', encoding='utf-8') as f:
            system_prompt = f.read()
        
        # 작은 샘플로 테스트
        test_samples = self.samples.head(sample_size)
        sentences = test_samples['user_prompt'].tolist()
        answers = test_samples['output'].tolist()
        
        # 한 번에 모든 문장 처리 (API 호출 최소화)
        input_text = system_prompt + "\n\n"
        for i, sentence in enumerate(sentences, 1):
            input_text += f"{i}. {sentence}\n"
        
        try:
            start_time = time.time()
            response = self.model.generate_content(input_text)
            response_text = response.text
            test_duration = time.time() - start_time
            
            # 결과 파싱
            predictions = self._parse_results(response_text, len(sentences))
            
            # 정확도 계산
            correct = sum(1 for p, a in zip(predictions, answers) if p == a)
            accuracy = correct / len(answers)
            
            # 속성별 정확도
            attr_acc = self._calculate_attribute_accuracy(predictions, answers)
            
            result = {
                "prompt_name": os.path.basename(prompt_path),
                "accuracy": accuracy,
                "correct": correct,
                "total": len(answers),
                "attribute_accuracies": attr_acc,
                "parsing_failures": sum(1 for p in predictions if not p),
                "duration": test_duration
            }
            
            print(f"정확도: {accuracy:.3f} ({correct}/{len(answers)})")
            print(f"속성별: 유형={attr_acc['type']:.2f}, 극성={attr_acc['polarity']:.2f}, 시제={attr_acc['tense']:.2f}, 확실성={attr_acc['certainty']:.2f}")
            
            return result
            
        except Exception as e:
            print(f"테스트 오류: {e}")
            return {
                "prompt_name": os.path.basename(prompt_path),
                "accuracy": 0.0,
                "error": str(e)
            }
    
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
    
    def _calculate_attribute_accuracy(self, predictions: List[str], answers: List[str]) -> Dict[str, float]:
        """속성별 정확도 계산"""
        total = len(answers)
        type_correct = polarity_correct = tense_correct = certainty_correct = 0
        
        for pred, ans in zip(predictions, answers):
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
        
        return {
            "type": type_correct / total,
            "polarity": polarity_correct / total,
            "tense": tense_correct / total,
            "certainty": certainty_correct / total
        }
    
    def test_best_prompt_only(self) -> Dict:
        """최고 성능 프롬프트만 테스트"""
        # 개선된 v2 프롬프트 테스트
        best_prompt_path = "prompt/gemini/enhanced_v3.txt"
        
        if os.path.exists(best_prompt_path):
            return self.test_single_prompt_efficiently(best_prompt_path, sample_size=30)
        else:
            print(f"프롬프트 파일을 찾을 수 없습니다: {best_prompt_path}")
            return {}

if __name__ == "__main__":
    try:
        tester = OptimizedGeminiTester()
        
        # 최고 성능 프롬프트만 테스트
        result = tester.test_best_prompt_only()
        
        if result and result.get('accuracy', 0) >= 0.7:
            print("\n🎉 0.7점 이상 달성! GPT-4o 최종 검증 준비 완료")
        elif result:
            print(f"\n📈 현재 성능: {result.get('accuracy', 0):.3f} - 추가 최적화 필요")
        
    except Exception as e:
        print(f"테스트 오류: {e}")