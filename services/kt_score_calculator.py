"""
KT 해커톤 점수 계산기

KT 점수 공식: 0.8 × 정확도 + 0.1 × 한글비율 + 0.1 × 길이점수
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class KTScoreBreakdown:
    """KT 점수 세부 분석 결과"""
    accuracy_score: float              # 정확도 점수 (0.8 가중치)
    korean_ratio_score: float          # 한글 비율 점수 (0.1 가중치)
    length_score: float                # 길이 점수 (0.1 가중치)
    total_score: float                 # 총점
    korean_char_ratio: float           # 실제 한글 문자 비율
    prompt_length: int                 # 프롬프트 길이
    improvement_suggestions: List[str]  # 개선 제안사항

class KTScoreCalculator:
    """KT 해커톤 점수 계산기"""
    
    def __init__(self, max_length: int = 3000):
        self.max_length = max_length
        self.accuracy_weight = 0.8
        self.korean_ratio_weight = 0.1
        self.length_weight = 0.1
        
    def calculate_total_score(self, accuracy: float, korean_ratio: float, length_score: float) -> float:
        """총점 계산"""
        total = (
            self.accuracy_weight * accuracy +
            self.korean_ratio_weight * korean_ratio +
            self.length_weight * length_score
        )
        return round(total, 4)
    
    def calculate_korean_ratio(self, prompt_text: str) -> float:
        """한글 문자 비율 계산 (공백·줄바꿈 제외)"""
        # 공백과 줄바꿈 제거
        text_no_spaces = re.sub(r'\s+', '', prompt_text)
        
        if not text_no_spaces:
            return 0.0
            
        # 한글 문자 개수 계산 (가-힣)
        korean_chars = len(re.findall(r'[가-힣]', text_no_spaces))
        total_chars = len(text_no_spaces)
        
        ratio = korean_chars / total_chars if total_chars > 0 else 0.0
        return round(ratio, 4)
    
    def calculate_length_score(self, prompt_length: int, max_length: int = None) -> float:
        """길이 점수 계산 (짧을수록 높은 점수)"""
        if max_length is None:
            max_length = self.max_length
            
        if prompt_length >= max_length:
            return 0.0
        
        # 길이가 짧을수록 높은 점수 (선형적으로 감소)
        score = 1.0 - (prompt_length / max_length)
        return round(max(0.0, score), 4)
    
    def calculate_full_score(self, accuracy: float, prompt_text: str) -> KTScoreBreakdown:
        """전체 점수 계산 및 분석"""
        korean_ratio = self.calculate_korean_ratio(prompt_text)
        prompt_length = len(prompt_text)
        length_score = self.calculate_length_score(prompt_length)
        
        # 가중치 적용된 개별 점수
        weighted_accuracy = self.accuracy_weight * accuracy
        weighted_korean = self.korean_ratio_weight * korean_ratio
        weighted_length = self.length_weight * length_score
        
        total_score = self.calculate_total_score(accuracy, korean_ratio, length_score)
        
        # 개선 제안사항 생성
        suggestions = self._generate_improvement_suggestions(
            accuracy, korean_ratio, length_score, prompt_length
        )
        
        return KTScoreBreakdown(
            accuracy_score=weighted_accuracy,
            korean_ratio_score=weighted_korean,
            length_score=weighted_length,
            total_score=total_score,
            korean_char_ratio=korean_ratio,
            prompt_length=prompt_length,
            improvement_suggestions=suggestions
        )
    
    def _generate_improvement_suggestions(
        self, accuracy: float, korean_ratio: float, length_score: float, prompt_length: int
    ) -> List[str]:
        """개선 제안사항 생성"""
        suggestions = []
        
        # 정확도 개선 (가장 중요)
        if accuracy < 0.8:
            suggestions.append(f"정확도 개선 필요: {accuracy:.3f} → 0.8+ (가중치 80%)")
        elif accuracy < 0.85:
            suggestions.append(f"정확도 추가 개선 가능: {accuracy:.3f} → 0.85+")
            
        # 한글 비율 개선
        if korean_ratio < 0.9:
            suggestions.append(f"한글 비율 개선 필요: {korean_ratio:.3f} → 0.9+ (영어→한글 번역)")
        
        # 길이 최적화
        if prompt_length > self.max_length * 0.8:
            suggestions.append(f"길이 압축 권장: {prompt_length}자 → {int(self.max_length * 0.7)}자 이하")
        elif prompt_length > self.max_length * 0.6:
            suggestions.append(f"길이 최적화 가능: {prompt_length}자 → 더 짧게")
            
        if not suggestions:
            suggestions.append("모든 지표가 우수합니다!")
            
        return suggestions
    
    def get_improvement_priority(self, current_scores: Dict[str, float]) -> str:
        """개선 우선순위 제시"""
        accuracy = current_scores.get('accuracy', 0.0)
        korean_ratio = current_scores.get('korean_ratio', 0.0)
        length_score = current_scores.get('length_score', 0.0)
        
        # 정확도가 가장 중요 (80% 가중치)
        if accuracy < 0.8:
            return "정확도 최우선 개선 (0.8점 이상 달성)"
        elif korean_ratio < 0.9:
            return "한글 비율 개선 (90% 이상 달성)"
        elif length_score < 0.5:
            return "길이 압축 최적화 (더 짧게)"
        else:
            return "모든 지표 균형 개선"
    
    def compare_scores(self, score1: KTScoreBreakdown, score2: KTScoreBreakdown) -> Dict[str, str]:
        """두 점수 비교 분석"""
        comparison = {}
        
        # 총점 비교
        total_diff = score2.total_score - score1.total_score
        comparison['total'] = f"총점: {score1.total_score:.4f} → {score2.total_score:.4f} ({total_diff:+.4f})"
        
        # 정확도 비교
        acc_diff = score2.accuracy_score - score1.accuracy_score
        comparison['accuracy'] = f"정확도: {score1.accuracy_score:.4f} → {score2.accuracy_score:.4f} ({acc_diff:+.4f})"
        
        # 한글 비율 비교
        kr_diff = score2.korean_ratio_score - score1.korean_ratio_score
        comparison['korean'] = f"한글비율: {score1.korean_ratio_score:.4f} → {score2.korean_ratio_score:.4f} ({kr_diff:+.4f})"
        
        # 길이 점수 비교
        len_diff = score2.length_score - score1.length_score
        comparison['length'] = f"길이점수: {score1.length_score:.4f} → {score2.length_score:.4f} ({len_diff:+.4f})"
        
        return comparison
    
    def format_score_report(self, score: KTScoreBreakdown) -> str:
        """점수 리포트 포맷팅"""
        report = f"""
=== KT 해커톤 점수 분석 ===
총점: {score.total_score:.4f}

구성 요소별 점수:
- 정확도 (80%): {score.accuracy_score:.4f}
- 한글비율 (10%): {score.korean_ratio_score:.4f} (실제 비율: {score.korean_char_ratio:.3f})
- 길이점수 (10%): {score.length_score:.4f} (길이: {score.prompt_length}자)

개선 제안사항:
"""
        for i, suggestion in enumerate(score.improvement_suggestions, 1):
            report += f"{i}. {suggestion}\n"
            
        return report.strip()