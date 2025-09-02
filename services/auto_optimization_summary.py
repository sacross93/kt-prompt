"""
자동화된 최적화 시스템 요약 및 리포트 생성기
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class AutoOptimizationSummary:
    """자동화된 최적화 결과 요약 및 리포트 생성"""
    
    def __init__(self, output_dir: str = "prompt/auto_optimized"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_comprehensive_report(
        self, 
        accuracy_history: List[Dict],
        length_optimization_info: Optional[Dict] = None,
        final_kt_score: float = 0.0
    ) -> str:
        """종합 최적화 리포트 생성"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report = f"""# 🤖 자동화된 프롬프트 최적화 리포트

생성 시간: {timestamp}

## 📊 최적화 개요

### 자동화된 정확도 최적화 (Gemini Pro 기반)
- **총 반복 수**: {len(accuracy_history)}회
- **시작 정확도**: {accuracy_history[0]['accuracy']:.4f} ({accuracy_history[0]['accuracy']*100:.1f}%)
- **최종 정확도**: {accuracy_history[-1]['accuracy']:.4f} ({accuracy_history[-1]['accuracy']*100:.1f}%)
- **정확도 개선**: {accuracy_history[-1]['accuracy'] - accuracy_history[0]['accuracy']:+.4f} ({(accuracy_history[-1]['accuracy'] - accuracy_history[0]['accuracy'])*100:+.1f}%p)

### 최종 KT 점수
- **총점**: {final_kt_score:.4f} / 1.0000
- **목표 달성**: {'✅ 달성' if final_kt_score >= 0.9 else '❌ 미달성'}

## 🔄 반복별 상세 진행 상황

"""
        
        for i, iteration in enumerate(accuracy_history, 1):
            accuracy_change = ""
            if i > 1:
                prev_accuracy = accuracy_history[i-2]['accuracy']
                change = iteration['accuracy'] - prev_accuracy
                accuracy_change = f" ({change:+.4f})"
            
            report += f"""### 반복 {i}
- **정확도**: {iteration['accuracy']:.4f}{accuracy_change}
- **오답 수**: {iteration.get('error_count', 'N/A')}개
- **KT 점수**: {iteration.get('kt_score', 'N/A'):.4f}
- **타임스탬프**: {iteration.get('timestamp', 'N/A')}

"""
            
            if iteration.get('improvements'):
                report += "**개선사항**:\n"
                for improvement in iteration['improvements']:
                    report += f"- {improvement}\n"
                report += "\n"
        
        if length_optimization_info:
            report += f"""## 📏 길이 압축 최적화

- **원본 길이**: {length_optimization_info.get('original_length', 'N/A')}자
- **압축 후 길이**: {length_optimization_info.get('compressed_length', 'N/A')}자
- **압축률**: {length_optimization_info.get('compression_ratio', 0)*100:.1f}%
- **정확도 변화**: {length_optimization_info.get('accuracy_change', 0):+.4f}
- **KT 점수 변화**: {length_optimization_info.get('kt_score_change', 0):+.4f}

"""
        
        # 성능 분석
        report += """## 📈 성능 분석

### 자동화 시스템의 장점
1. **지능적 오답 분석**: Gemini 2.5 Pro가 오답 패턴을 체계적으로 분석
2. **맞춤형 개선**: 각 반복마다 구체적인 개선 방향 제시
3. **자동화된 반복**: 목표 달성까지 자동으로 반복 수행
4. **실시간 모니터링**: 각 단계별 성능 변화 추적

### 개선 효과
"""
        
        if len(accuracy_history) > 1:
            total_improvement = accuracy_history[-1]['accuracy'] - accuracy_history[0]['accuracy']
            avg_improvement_per_iteration = total_improvement / (len(accuracy_history) - 1)
            
            report += f"""- **평균 반복당 개선**: {avg_improvement_per_iteration:+.4f}
- **총 개선 폭**: {total_improvement:+.4f}
- **개선 효율성**: {'높음' if avg_improvement_per_iteration > 0.01 else '보통' if avg_improvement_per_iteration > 0 else '낮음'}

"""
        
        # 추천사항
        report += """## 💡 추가 개선 권장사항

### 단기 개선
1. **API 할당량 관리**: Gemini Pro 사용량 최적화
2. **배치 크기 조정**: 테스트 샘플 수 조정으로 속도 향상
3. **캐싱 활용**: 중복 요청 방지로 효율성 증대

### 장기 개선
1. **하이브리드 접근**: Rule-based + AI 기반 최적화 결합
2. **도메인 특화**: KT 해커톤 데이터 특성에 맞춘 전용 최적화
3. **앙상블 방법**: 여러 모델의 결과를 조합한 최적화

## 🎯 결론

"""
        
        if final_kt_score >= 0.9:
            report += """✅ **목표 달성**: 자동화된 최적화 시스템이 성공적으로 KT 점수 0.9점 이상을 달성했습니다.

**핵심 성공 요인**:
- Gemini 2.5 Pro의 지능적 분석 능력
- 체계적인 반복 최적화 프로세스
- 실시간 성능 모니터링

"""
        else:
            needed_improvement = 0.9 - final_kt_score
            report += f"""❌ **목표 미달**: 현재 점수에서 {needed_improvement:.4f}점 추가 개선이 필요합니다.

**추가 최적화 방안**:
- 더 많은 반복 수행 (현재 {len(accuracy_history)}회)
- 다양한 최적화 전략 시도
- 데이터 품질 개선

"""
        
        report += """
---
*이 리포트는 자동화된 프롬프트 최적화 시스템에 의해 생성되었습니다.*
"""
        
        return report
    
    def save_optimization_artifacts(
        self,
        final_prompt: str,
        accuracy_history: List[Dict],
        length_info: Optional[Dict] = None,
        final_kt_score: float = 0.0
    ) -> Dict[str, str]:
        """최적화 결과물 저장"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifacts = {}
        
        # 1. 최종 프롬프트 저장
        final_prompt_path = os.path.join(self.output_dir, f"final_optimized_{timestamp}.txt")
        with open(final_prompt_path, 'w', encoding='utf-8') as f:
            f.write(final_prompt)
        artifacts['final_prompt'] = final_prompt_path
        
        # 2. 최적화 히스토리 저장
        history_path = os.path.join(self.output_dir, f"optimization_history_{timestamp}.json")
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy_history': accuracy_history,
                'length_optimization': length_info,
                'final_kt_score': final_kt_score,
                'timestamp': timestamp
            }, f, ensure_ascii=False, indent=2)
        artifacts['history'] = history_path
        
        # 3. 종합 리포트 저장
        report_content = self.generate_comprehensive_report(accuracy_history, length_info, final_kt_score)
        report_path = os.path.join(self.output_dir, f"optimization_report_{timestamp}.md")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        artifacts['report'] = report_path
        
        # 4. 요약 정보 저장
        summary_path = os.path.join(self.output_dir, f"summary_{timestamp}.json")
        summary_data = {
            'timestamp': timestamp,
            'final_kt_score': final_kt_score,
            'total_iterations': len(accuracy_history),
            'accuracy_improvement': accuracy_history[-1]['accuracy'] - accuracy_history[0]['accuracy'] if accuracy_history else 0,
            'goal_achieved': final_kt_score >= 0.9,
            'artifacts': artifacts
        }
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        artifacts['summary'] = summary_path
        
        logger.info(f"최적화 결과물 저장 완료: {len(artifacts)}개 파일")
        return artifacts
    
    def create_performance_visualization(self, accuracy_history: List[Dict]) -> str:
        """성능 변화 시각화 (텍스트 기반)"""
        
        if not accuracy_history:
            return "시각화할 데이터가 없습니다."
        
        visualization = "\n📊 정확도 변화 시각화\n"
        visualization += "=" * 50 + "\n"
        
        max_accuracy = max(h['accuracy'] for h in accuracy_history)
        min_accuracy = min(h['accuracy'] for h in accuracy_history)
        
        for i, history in enumerate(accuracy_history, 1):
            accuracy = history['accuracy']
            
            # 진행률 바 생성 (0-100% 기준)
            bar_length = 30
            filled_length = int(bar_length * accuracy)
            bar = "█" * filled_length + "░" * (bar_length - filled_length)
            
            # 변화 표시
            change_indicator = ""
            if i > 1:
                prev_accuracy = accuracy_history[i-2]['accuracy']
                if accuracy > prev_accuracy:
                    change_indicator = " ↗️"
                elif accuracy < prev_accuracy:
                    change_indicator = " ↘️"
                else:
                    change_indicator = " ➡️"
            
            visualization += f"반복 {i:2d}: [{bar}] {accuracy:.4f}{change_indicator}\n"
        
        # 통계 정보
        visualization += "\n📈 통계 정보\n"
        visualization += "-" * 30 + "\n"
        visualization += f"최고 정확도: {max_accuracy:.4f}\n"
        visualization += f"최저 정확도: {min_accuracy:.4f}\n"
        visualization += f"평균 정확도: {sum(h['accuracy'] for h in accuracy_history) / len(accuracy_history):.4f}\n"
        visualization += f"총 개선폭: {accuracy_history[-1]['accuracy'] - accuracy_history[0]['accuracy']:+.4f}\n"
        
        return visualization