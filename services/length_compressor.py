"""
길이 압축 최적화기

프롬프트를 점진적으로 압축하면서 핵심 내용을 보존
"""

import re
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class CompressionResult:
    """압축 결과"""
    original_text: str
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    removed_redundancies: List[str]    # 제거된 중복 표현들
    shortened_sentences: List[str]     # 단축된 문장들
    preserved_concepts: List[str]      # 보존된 핵심 개념들
    quality_score: float               # 압축 품질 점수

class LengthCompressor:
    """길이 압축 최적화기"""
    
    def __init__(self):
        # 제거 가능한 중복 표현들
        self.redundant_patterns = [
            r'\s+',  # 연속 공백
            r'\n\s*\n\s*\n+',  # 연속 줄바꿈
            r'(?:그리고|또한|더불어|아울러)\s*,?\s*(?:그리고|또한|더불어|아울러)',  # 중복 접속사
            r'(?:예를 들어|예시로)\s*,?\s*(?:예를 들어|예시로)',  # 중복 예시 표현
            r'(?:다음과 같이|아래와 같이)\s*,?\s*(?:다음과 같이|아래와 같이)',  # 중복 지시 표현
        ]
        
        # 단축 가능한 표현들
        self.shortening_rules = {
            r'다음과 같이 분류해주세요': '분류하세요',
            r'아래와 같은 형식으로': '다음 형식으로',
            r'반드시 준수해야 합니다': '준수하세요',
            r'주의깊게 살펴보고': '확인하고',
            r'정확하게 판단하여': '판단하여',
            r'올바르게 분류하세요': '분류하세요',
            r'다음 중에서 선택하세요': '선택하세요',
            r'반드시 포함되어야 합니다': '포함하세요',
            r'다음과 같은 규칙을 따르세요': '규칙을 따르세요',
            r'아래 예시를 참고하세요': '예시 참고',
            r'정확한 답변을 제공해주세요': '정확히 답변하세요',
        }
        
    def remove_redundancy(self, prompt: str) -> str:
        """중복 표현 제거"""
        result = prompt
        removed = []
        
        for pattern in self.redundant_patterns:
            matches = re.findall(pattern, result)
            if matches:
                if pattern == r'\s+':
                    result = re.sub(pattern, ' ', result)
                elif pattern == r'\n\s*\n\s*\n+':
                    result = re.sub(pattern, '\n\n', result)
                else:
                    result = re.sub(pattern, lambda m: m.group().split()[0], result)
                    removed.extend(matches)
        
        if removed:
            logger.info(f"제거된 중복 표현: {len(removed)}개")
            
        return result.strip()
    
    def compress_sentences(self, prompt: str) -> str:
        """문장 압축"""
        result = prompt
        shortened = []
        
        for pattern, replacement in self.shortening_rules.items():
            if re.search(pattern, result):
                result = re.sub(pattern, replacement, result)
                shortened.append(f"{pattern} → {replacement}")
        
        if shortened:
            logger.info(f"단축된 표현: {len(shortened)}개")
            
        return result
    
    def preserve_core_content(self, original: str, compressed: str) -> bool:
        """핵심 내용 보존 여부 확인"""
        # 핵심 키워드들이 보존되었는지 확인
        core_keywords = [
            '분류', '유형', '극성', '시제', '확실성',
            '긍정', '부정', '중립', '현재', '과거', '미래',
            '확실', '불확실', '예시', '형식', '출력'
        ]
        
        preserved_count = 0
        total_keywords = len(core_keywords)
        
        for keyword in core_keywords:
            if keyword in original and keyword in compressed:
                preserved_count += 1
        
        preservation_ratio = preserved_count / total_keywords
        return preservation_ratio >= 0.8  # 80% 이상 보존
    
    async def optimize_information_density(self, prompt: str, max_length: int) -> str:
        """정보 밀도 최적화"""
        current_length = len(prompt)
        logger.info(f"압축 시작: {current_length}자 → {max_length}자 목표")
        
        if current_length <= max_length:
            logger.info("이미 목표 길이 이하")
            return prompt
        
        # 1단계: 중복 제거
        step1 = self.remove_redundancy(prompt)
        length1 = len(step1)
        logger.info(f"1단계 중복 제거: {length1}자 ({current_length - length1}자 단축)")
        
        if length1 <= max_length:
            return step1
        
        # 2단계: 문장 압축
        step2 = self.compress_sentences(step1)
        length2 = len(step2)
        logger.info(f"2단계 문장 압축: {length2}자 ({length1 - length2}자 단축)")
        
        if length2 <= max_length:
            return step2
        
        # 3단계: 고급 압축 (예시 줄이기, 설명 간소화)
        step3 = self._advanced_compression(step2, max_length)
        length3 = len(step3)
        logger.info(f"3단계 고급 압축: {length3}자 ({length2 - length3}자 단축)")
        
        # 핵심 내용 보존 확인
        if not self.preserve_core_content(prompt, step3):
            logger.warning("핵심 내용 손실 위험 - 이전 단계 결과 사용")
            return step2 if self.preserve_core_content(prompt, step2) else step1
        
        return step3
    
    def _advanced_compression(self, prompt: str, max_length: int) -> str:
        """고급 압축 기법"""
        result = prompt
        current_length = len(result)
        
        if current_length <= max_length:
            return result
        
        # 예시 개수 줄이기
        result = self._reduce_examples(result)
        
        # 설명 간소화
        result = self._simplify_explanations(result)
        
        # 불필요한 수식어 제거
        result = self._remove_modifiers(result)
        
        return result
    
    def _reduce_examples(self, prompt: str) -> str:
        """예시 개수 줄이기"""
        # 예시 섹션 찾기
        example_pattern = r'예시\s*\d*\s*:.*?(?=예시\s*\d*\s*:|$)'
        examples = re.findall(example_pattern, prompt, re.DOTALL)
        
        if len(examples) > 3:  # 3개 이상이면 줄이기
            # 가장 대표적인 예시들만 유지
            selected_examples = examples[:2] + examples[-1:]  # 처음 2개 + 마지막 1개
            
            # 원본에서 예시 부분 교체
            for i, example in enumerate(examples):
                if i < 2 or i == len(examples) - 1:
                    continue
                prompt = prompt.replace(example, '')
        
        return prompt
    
    def _simplify_explanations(self, prompt: str) -> str:
        """설명 간소화"""
        simplifications = {
            r'매우\s+중요합니다': '중요합니다',
            r'반드시\s+필요합니다': '필요합니다',
            r'정확하게\s+': '',
            r'올바르게\s+': '',
            r'주의깊게\s+': '',
            r'세심하게\s+': '',
            r'철저하게\s+': '',
            r'완전히\s+': '',
            r'확실히\s+': '',
        }
        
        result = prompt
        for pattern, replacement in simplifications.items():
            result = re.sub(pattern, replacement, result)
        
        return result
    
    def _remove_modifiers(self, prompt: str) -> str:
        """불필요한 수식어 제거"""
        # 과도한 형용사/부사 제거
        modifiers_to_remove = [
            r'매우\s+', r'정말\s+', r'아주\s+', r'상당히\s+',
            r'꽤\s+', r'상당한\s+', r'충분한\s+', r'적절한\s+'
        ]
        
        result = prompt
        for modifier in modifiers_to_remove:
            result = re.sub(modifier, '', result)
        
        return result
    
    def validate_compression_quality(self, original: str, compressed: str) -> CompressionResult:
        """압축 품질 검증"""
        original_length = len(original)
        compressed_length = len(compressed)
        compression_ratio = compressed_length / original_length if original_length > 0 else 0
        
        # 핵심 개념 보존 확인
        preserved_concepts = self._extract_preserved_concepts(original, compressed)
        
        # 품질 점수 계산 (0-1)
        preservation_score = len(preserved_concepts) / 10  # 가정: 10개 핵심 개념
        length_score = 1 - compression_ratio  # 압축률이 높을수록 좋음
        quality_score = (preservation_score * 0.7 + length_score * 0.3)
        
        return CompressionResult(
            original_text=original,
            compressed_text=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=compression_ratio,
            removed_redundancies=[],  # 실제 구현에서는 추적
            shortened_sentences=[],   # 실제 구현에서는 추적
            preserved_concepts=preserved_concepts,
            quality_score=quality_score
        )
    
    def _extract_preserved_concepts(self, original: str, compressed: str) -> List[str]:
        """보존된 핵심 개념 추출"""
        core_concepts = [
            '문장 분류', '유형 판단', '극성 분석', '시제 확인', '확실성 평가',
            '긍정적', '부정적', '중립적', '현재시제', '과거시제', '미래시제',
            '확실함', '불확실함', '출력 형식', '예시 참고', '규칙 준수'
        ]
        
        preserved = []
        for concept in core_concepts:
            if concept in original and concept in compressed:
                preserved.append(concept)
        
        return preserved