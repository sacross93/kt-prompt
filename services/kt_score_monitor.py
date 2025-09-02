"""
KT 점수 실시간 모니터링 시스템

각 최적화 단계별 KT 점수 변화 추이를 실시간으로 추적하고 시각화
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import logging

from .kt_score_calculator import KTScoreBreakdown

logger = logging.getLogger(__name__)

@dataclass
class ScoreHistory:
    """점수 변화 히스토리"""
    timestamp: str
    phase: str
    total_score: float
    accuracy_score: float
    korean_ratio_score: float
    length_score: float
    accuracy_raw: float
    korean_ratio_raw: float
    prompt_length: int
    improvements: List[str]

class KTScoreMonitor:
    """KT 점수 실시간 모니터링 시스템"""
    
    def __init__(self, output_dir: str = "analysis"):
        self.output_dir = output_dir
        self.history_file = os.path.join(output_dir, "kt_score_history.json")
        self.score_history: List[ScoreHistory] = []
        
        # 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        
        # 기존 히스토리 로드
        self._load_history()
        
    def _load_history(self):
        """기존 히스토리 로드"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.score_history = [
                        ScoreHistory(**item) for item in data
                    ]
                logger.info(f"기존 히스토리 로드: {len(self.score_history)}개 기록")
        except Exception as e:
            logger.warning(f"히스토리 로드 실패: {e}")
            self.score_history = []
    
    def _save_history(self):
        """히스토리 저장"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(item) for item in self.score_history], f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"히스토리 저장 실패: {e}")
    
    def record_score(self, phase: str, score_breakdown: KTScoreBreakdown):
        """점수 기록"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 개선 사항 분석
        improvements = self._analyze_improvements(score_breakdown)
        
        history_entry = ScoreHistory(
            timestamp=timestamp,
            phase=phase,
            total_score=score_breakdown.total_score,
            accuracy_score=score_breakdown.accuracy_score,
            korean_ratio_score=score_breakdown.korean_ratio_score,
            length_score=score_breakdown.length_score,
            accuracy_raw=score_breakdown.accuracy_score / 0.8,  # 원본 정확도
            korean_ratio_raw=score_breakdown.korean_char_ratio,
            prompt_length=score_breakdown.prompt_length,
            improvements=improvements
        )
        
        self.score_history.append(history_entry)
        self._save_history()
        
        logger.info(f"점수 기록 완료: {phase} - {score_breakdown.total_score:.4f}")
    
    def _analyze_improvements(self, current_score: KTScoreBreakdown) -> List[str]:
        """개선 사항 분석"""
        improvements = []
        
        if len(self.score_history) > 0:
            last_score = self.score_history[-1]
            
            # 총점 개선
            total_diff = current_score.total_score - last_score.total_score
            if total_diff > 0.001:
                improvements.append(f"총점 {total_diff:+.4f} 개선")
            elif total_diff < -0.001:
                improvements.append(f"총점 {total_diff:+.4f} 하락")
            
            # 정확도 개선
            acc_diff = current_score.accuracy_score - last_score.accuracy_score
            if acc_diff > 0.001:
                improvements.append(f"정확도 {acc_diff:+.4f} 개선")
            elif acc_diff < -0.001:
                improvements.append(f"정확도 {acc_diff:+.4f} 하락")
            
            # 한글 비율 개선
            kr_diff = current_score.korean_ratio_score - last_score.korean_ratio_score
            if kr_diff > 0.001:
                improvements.append(f"한글비율 {kr_diff:+.4f} 개선")
            elif kr_diff < -0.001:
                improvements.append(f"한글비율 {kr_diff:+.4f} 하락")
            
            # 길이 점수 개선
            len_diff = current_score.length_score - last_score.length_score
            if len_diff > 0.001:
                improvements.append(f"길이점수 {len_diff:+.4f} 개선")
            elif len_diff < -0.001:
                improvements.append(f"길이점수 {len_diff:+.4f} 하락")
        
        return improvements
    
    def get_progress_summary(self) -> Dict:
        """진행 상황 요약"""
        if not self.score_history:
            return {"message": "기록된 점수가 없습니다."}
        
        first_score = self.score_history[0]
        latest_score = self.score_history[-1]
        
        total_improvement = latest_score.total_score - first_score.total_score
        
        summary = {
            "total_records": len(self.score_history),
            "first_score": first_score.total_score,
            "latest_score": latest_score.total_score,
            "total_improvement": total_improvement,
            "current_phase": latest_score.phase,
            "target_score": 0.9,
            "remaining_gap": 0.9 - latest_score.total_score,
            "progress_percentage": (latest_score.total_score / 0.9) * 100
        }
        
        return summary
    
    def get_improvement_priority(self) -> str:
        """개선 우선순위 제시"""
        if not self.score_history:
            return "데이터 부족"
        
        latest = self.score_history[-1]
        
        # 각 구성 요소의 기여도 분석
        accuracy_contribution = latest.accuracy_score / 0.8  # 정확도의 실제 값
        korean_contribution = latest.korean_ratio_raw
        length_contribution = latest.length_score
        
        priorities = []
        
        if accuracy_contribution < 0.8:
            priorities.append(f"정확도 최우선 (현재: {accuracy_contribution:.3f}, 목표: 0.8+)")
        
        if korean_contribution < 0.9:
            priorities.append(f"한글비율 개선 (현재: {korean_contribution:.3f}, 목표: 0.9+)")
        
        if length_contribution < 0.5:
            priorities.append(f"길이 최적화 (현재 길이: {latest.prompt_length}자)")
        
        if not priorities:
            return "모든 지표가 목표 수준에 근접"
        
        return " → ".join(priorities)
    
    def generate_progress_report(self) -> str:
        """진행 상황 리포트 생성"""
        summary = self.get_progress_summary()
        
        if "message" in summary:
            return summary["message"]
        
        report = f"""
# KT 점수 모니터링 리포트

## 전체 진행 상황
- 총 기록 수: {summary['total_records']}개
- 시작 점수: {summary['first_score']:.4f}
- 현재 점수: {summary['latest_score']:.4f}
- 총 개선도: {summary['total_improvement']:+.4f}
- 현재 단계: {summary['current_phase']}

## 목표 달성 현황
- 목표 점수: {summary['target_score']:.1f}점
- 남은 격차: {summary['remaining_gap']:.4f}점
- 진행률: {summary['progress_percentage']:.1f}%

## 개선 우선순위
{self.get_improvement_priority()}

## 최근 변화 추이
"""
        
        # 최근 5개 기록 표시
        recent_records = self.score_history[-5:]
        for record in recent_records:
            report += f"\n### {record.phase} ({record.timestamp})\n"
            report += f"- 총점: {record.total_score:.4f}\n"
            report += f"- 정확도: {record.accuracy_raw:.3f} (가중치: {record.accuracy_score:.4f})\n"
            report += f"- 한글비율: {record.korean_ratio_raw:.3f} (가중치: {record.korean_ratio_score:.4f})\n"
            report += f"- 길이: {record.prompt_length}자 (점수: {record.length_score:.4f})\n"
            
            if record.improvements:
                report += f"- 개선사항: {', '.join(record.improvements)}\n"
        
        return report.strip()
    
    def create_score_visualization(self) -> str:
        """점수 변화 시각화 (텍스트 기반)"""
        if len(self.score_history) < 2:
            return "시각화를 위한 데이터가 부족합니다."
        
        # 간단한 텍스트 기반 그래프
        visualization = "\n=== KT 점수 변화 추이 ===\n\n"
        
        max_score = max(record.total_score for record in self.score_history)
        min_score = min(record.total_score for record in self.score_history)
        
        for i, record in enumerate(self.score_history):
            # 점수를 0-50 범위로 정규화
            normalized = int((record.total_score - min_score) / (max_score - min_score + 0.001) * 50)
            
            bar = "█" * normalized + "░" * (50 - normalized)
            visualization += f"{i+1:2d}. {record.phase:15s} │{bar}│ {record.total_score:.4f}\n"
        
        visualization += f"\n범위: {min_score:.4f} ~ {max_score:.4f}\n"
        visualization += f"목표: 0.9000\n"
        
        return visualization
    
    def export_data(self, format: str = "json") -> str:
        """데이터 내보내기"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if format == "json":
            export_file = os.path.join(self.output_dir, f"kt_score_export_{timestamp}.json")
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump([asdict(item) for item in self.score_history], f, 
                         ensure_ascii=False, indent=2)
        
        elif format == "csv":
            import csv
            export_file = os.path.join(self.output_dir, f"kt_score_export_{timestamp}.csv")
            with open(export_file, 'w', newline='', encoding='utf-8') as f:
                if self.score_history:
                    writer = csv.DictWriter(f, fieldnames=asdict(self.score_history[0]).keys())
                    writer.writeheader()
                    for record in self.score_history:
                        writer.writerow(asdict(record))
        
        return export_file