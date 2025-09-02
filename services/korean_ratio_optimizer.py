"""
한글 비율 최적화기

영어 용어를 한글로 번역하고 한글 비율을 90% 이상으로 최적화
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class KoreanOptimizationResult:
    """한글 최적화 결과"""
    original_text: str
    optimized_text: str
    original_korean_ratio: float
    optimized_korean_ratio: float
    translation_map: Dict[str, str]    # 영어→한글 번역 매핑
    replaced_terms: List[str]          # 대체된 용어들
    added_explanations: List[str]      # 추가된 한글 설명들

class KoreanRatioOptimizer:
    """한글 비율 최적화기"""
    
    def __init__(self):
        # 기술 용어 한글 번역 사전
        self.tech_translations = {
            # AI/ML 용어
            "accuracy": "정확도",
            "precision": "정밀도", 
            "recall": "재현율",
            "classification": "분류",
            "model": "모델",
            "training": "훈련",
            "testing": "테스트",
            "validation": "검증",
            "dataset": "데이터셋",
            "feature": "특성",
            "label": "레이블",
            "prediction": "예측",
            "performance": "성능",
            "optimization": "최적화",
            "algorithm": "알고리즘",
            
            # 프롬프트 엔지니어링 용어
            "prompt": "프롬프트",
            "few-shot": "퓨샷",
            "chain-of-thought": "사고연쇄",
            "reasoning": "추론",
            "instruction": "지시사항",
            "example": "예시",
            "template": "템플릿",
            "format": "형식",
            "output": "출력",
            "input": "입력",
            
            # 일반 기술 용어
            "system": "시스템",
            "process": "과정",
            "method": "방법",
            "approach": "접근법",
            "strategy": "전략",
            "technique": "기법",
            "framework": "프레임워크",
            "pipeline": "파이프라인",
            "workflow": "작업흐름",
            "parameter": "매개변수"
        }
        
    def calculate_korean_ratio(self, text: str) -> float:
        """한글 문자 비율 계산"""
        # 공백과 줄바꿈 제거
        text_no_spaces = re.sub(r'\s+', '', text)
        
        if not text_no_spaces:
            return 0.0
            
        # 한글 문자 개수 계산
        korean_chars = len(re.findall(r'[가-힣]', text_no_spaces))
        total_chars = len(text_no_spaces)
        
        return korean_chars / total_chars if total_chars > 0 else 0.0  
  
    def translate_english_terms(self, prompt: str) -> str:
        """영어 용어를 한글로 번역"""
        translated = prompt
        translation_log = []
        
        for english, korean in self.tech_translations.items():
            # 단어 경계를 고려한 대체
            pattern = r'\b' + re.escape(english) + r'\b'
            if re.search(pattern, translated, re.IGNORECASE):
                translated = re.sub(pattern, korean, translated, flags=re.IGNORECASE)
                translation_log.append(f"{english} → {korean}")
        
        if translation_log:
            logger.info(f"번역된 용어: {', '.join(translation_log)}")
            
        return translated
    
    def replace_technical_terms(self, prompt: str) -> str:
        """기술 용어를 한글 우선으로 변경"""
        # 영어(한글) 형태를 한글(영어) 형태로 변경
        pattern = r'([A-Za-z\-]+)\s*\(([가-힣]+)\)'
        
        def replace_func(match):
            english = match.group(1)
            korean = match.group(2)
            return f"{korean}({english})"
        
        result = re.sub(pattern, replace_func, prompt)
        
        # 단독 영어 용어를 한글로 대체
        result = self.translate_english_terms(result)
        
        return result
    
    def add_korean_explanations(self, prompt: str) -> str:
        """영어 용어에 한글 설명 추가"""
        # 번역되지 않은 영어 용어 찾기
        english_terms = re.findall(r'\b[A-Za-z]{3,}\b', prompt)
        
        explanations_added = []
        result = prompt
        
        for term in english_terms:
            if term.lower() in self.tech_translations:
                korean = self.tech_translations[term.lower()]
                # 이미 설명이 있는지 확인
                if f"{term}({korean})" not in result and f"{korean}({term})" not in result:
                    result = result.replace(term, f"{korean}({term})", 1)
                    explanations_added.append(f"{term} → {korean}({term})")
        
        if explanations_added:
            logger.info(f"설명 추가: {', '.join(explanations_added)}")
            
        return result
    
    def optimize_to_target_ratio(self, prompt: str, target_ratio: float = 0.9) -> str:
        """목표 한글 비율까지 최적화"""
        current_ratio = self.calculate_korean_ratio(prompt)
        logger.info(f"현재 한글 비율: {current_ratio:.4f}, 목표: {target_ratio:.4f}")
        
        if current_ratio >= target_ratio:
            return prompt
        
        # 1단계: 기술 용어 번역
        step1 = self.translate_english_terms(prompt)
        ratio1 = self.calculate_korean_ratio(step1)
        logger.info(f"1단계 번역 후: {ratio1:.4f}")
        
        if ratio1 >= target_ratio:
            return step1
        
        # 2단계: 기술 용어 한글 우선으로 변경
        step2 = self.replace_technical_terms(step1)
        ratio2 = self.calculate_korean_ratio(step2)
        logger.info(f"2단계 용어 변경 후: {ratio2:.4f}")
        
        if ratio2 >= target_ratio:
            return step2
        
        # 3단계: 영어 용어에 한글 설명 추가
        step3 = self.add_korean_explanations(step2)
        ratio3 = self.calculate_korean_ratio(step3)
        logger.info(f"3단계 설명 추가 후: {ratio3:.4f}")
        
        if ratio3 >= target_ratio:
            return step3
        
        # 4단계: 추가 한글화 (영어 문장을 한글로)
        step4 = self._enhance_korean_content(step3)
        ratio4 = self.calculate_korean_ratio(step4)
        logger.info(f"4단계 추가 한글화 후: {ratio4:.4f}")
        
        return step4
    
    def _enhance_korean_content(self, prompt: str) -> str:
        """추가 한글 콘텐츠 강화"""
        # 영어 문장 패턴을 한글로 변환
        replacements = {
            r'\bPlease\b': '다음과 같이',
            r'\bNote that\b': '주의하세요',
            r'\bFor example\b': '예를 들어',
            r'\bIn other words\b': '다시 말해',
            r'\bTherefore\b': '따라서',
            r'\bHowever\b': '하지만',
            r'\bMoreover\b': '또한',
            r'\bFinally\b': '마지막으로',
            r'\bFirst\b': '첫째',
            r'\bSecond\b': '둘째',
            r'\bThird\b': '셋째',
            r'\bStep \d+': lambda m: f"{m.group().split()[1]}단계",
        }
        
        result = prompt
        for pattern, replacement in replacements.items():
            if callable(replacement):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
            else:
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        
        return result
    
    def get_optimization_result(self, original: str, optimized: str) -> KoreanOptimizationResult:
        """최적화 결과 분석"""
        original_ratio = self.calculate_korean_ratio(original)
        optimized_ratio = self.calculate_korean_ratio(optimized)
        
        # 변경된 용어들 추출
        translation_map = {}
        replaced_terms = []
        added_explanations = []
        
        # 간단한 변경 사항 추적 (실제로는 더 정교한 diff 알고리즘 필요)
        for english, korean in self.tech_translations.items():
            if english in original and korean in optimized:
                translation_map[english] = korean
                replaced_terms.append(english)
        
        return KoreanOptimizationResult(
            original_text=original,
            optimized_text=optimized,
            original_korean_ratio=original_ratio,
            optimized_korean_ratio=optimized_ratio,
            translation_map=translation_map,
            replaced_terms=replaced_terms,
            added_explanations=added_explanations
        )