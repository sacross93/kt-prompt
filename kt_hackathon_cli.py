"""
KT 해커톤 특화 CLI

3단계 프롬프트 최적화 시스템
1단계: 정확도 최우선 (0.8+)
2단계: 한글 비율 최적화 (90%+)  
3단계: 길이 압축 (3000자 이하)
"""

import argparse
import asyncio
import sys
import os
from typing import Optional
import json
from datetime import datetime

from services.kt_score_calculator import KTScoreCalculator
from services.three_phase_optimizer import ThreePhaseOptimizer
from services.samples_data_processor import SamplesDataProcessor
from config import OptimizationConfig
from utils.logging_utils import setup_logging

def print_kt_banner():
    """KT 해커톤 배너 출력"""
    print("🏆 KT 해커톤 프롬프트 최적화 시스템")
    print("=" * 60)
    print("📊 KT 점수 공식: 0.8×정확도 + 0.1×한글비율 + 0.1×길이점수")
    print("🎯 목표: 총점 0.9점 이상 달성")
    print("=" * 60)
    print()

async def run_kt_optimization(
    initial_prompt_path: str,
    samples_csv_path: str = "data/samples.csv",
    output_dir: str = "prompt/gemini",
    target_score: float = 0.9,
    use_auto: bool = False
) -> int:
    """KT 해커톤 3단계 최적화 실행"""
    
    try:
        print_kt_banner()
        
        # 초기 프롬프트 로드
        if not os.path.exists(initial_prompt_path):
            print(f"❌ 초기 프롬프트 파일을 찾을 수 없습니다: {initial_prompt_path}")
            return 1
            
        with open(initial_prompt_path, 'r', encoding='utf-8') as f:
            initial_prompt = f.read().strip()
        
        print(f"📄 초기 프롬프트 로드: {initial_prompt_path}")
        print(f"📊 샘플 데이터: {samples_csv_path}")
        print(f"🎯 목표 KT 점수: {target_score}")
        print()
        
        # 3단계 최적화기 초기화
        optimizer = ThreePhaseOptimizer(samples_csv_path)
        
        # 최적화 실행
        if use_auto:
            print("🤖 자동화된 3단계 최적화 시작... (Gemini Pro 기반)")
            result = await optimizer.execute_automated_optimization(initial_prompt)
        else:
            print("🚀 기본 3단계 최적화 시작...")
            result = await optimizer.execute_full_optimization(initial_prompt)
        
        # 결과 저장
        os.makedirs(output_dir, exist_ok=True)
        
        # 각 단계별 프롬프트 저장
        phase1_path = os.path.join(output_dir, "kt_phase1_accuracy.txt")
        phase2_path = os.path.join(output_dir, "kt_phase2_korean.txt")
        phase3_path = os.path.join(output_dir, "kt_phase3_final.txt")
        
        with open(phase1_path, 'w', encoding='utf-8') as f:
            f.write(result.phase1_prompt)
        
        with open(phase2_path, 'w', encoding='utf-8') as f:
            f.write(result.phase2_prompt)
            
        with open(phase3_path, 'w', encoding='utf-8') as f:
            f.write(result.phase3_prompt)
        
        # 리포트 저장
        report = optimizer.get_optimization_report(result)
        report_path = os.path.join(output_dir, "kt_optimization_report.md")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 결과 출력
        print("\n" + "="*60)
        print("🎉 KT 해커톤 최적화 완료!")
        print("="*60)
        
        final_score = result.final_kt_score
        print(f"🏆 최종 KT 점수: {final_score:.4f}")
        
        if final_score >= target_score:
            print("✅ 목표 점수 달성!")
        else:
            needed = target_score - final_score
            print(f"❌ 목표까지 {needed:.4f}점 부족")
        
        print(f"\n📁 결과 파일:")
        print(f"  - 1단계 (정확도): {phase1_path}")
        print(f"  - 2단계 (한글화): {phase2_path}")
        print(f"  - 3단계 (최종): {phase3_path}")
        print(f"  - 리포트: {report_path}")
        
        return 0 if final_score >= target_score else 1
        
    except Exception as e:
        print(f"❌ 최적화 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1

def calculate_kt_score(prompt_path: str, accuracy: float) -> None:
    """KT 점수 계산"""
    try:
        if not os.path.exists(prompt_path):
            print(f"❌ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        calculator = KTScoreCalculator()
        score_breakdown = calculator.calculate_full_score(accuracy, prompt_text)
        
        print("\n" + "="*50)
        print("📊 KT 점수 계산 결과")
        print("="*50)
        print(calculator.format_score_report(score_breakdown))
        
    except Exception as e:
        print(f"❌ 점수 계산 실패: {e}")

def analyze_samples_data(csv_path: str = "data/samples.csv") -> None:
    """샘플 데이터 분석"""
    try:
        processor = SamplesDataProcessor(csv_path)
        analysis = processor.analyze_samples_csv()
        
        print("\n" + "="*50)
        print("📊 샘플 데이터 분석 결과")
        print("="*50)
        print(processor.get_analysis_report())
        
    except Exception as e:
        print(f"❌ 데이터 분석 실패: {e}")

def test_prompt_accuracy(prompt_path: str, samples_csv_path: str = "data/samples.csv") -> None:
    """프롬프트 정확도 테스트"""
    try:
        from services.gemini_flash_classifier import GeminiFlashClassifier
        from config import OptimizationConfig
        
        if not os.path.exists(prompt_path):
            print(f"❌ 프롬프트 파일을 찾을 수 없습니다: {prompt_path}")
            return
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt_text = f.read().strip()
        
        print(f"🧪 프롬프트 정확도 테스트 시작...")
        print(f"📄 프롬프트: {prompt_path}")
        print(f"📊 데이터: {samples_csv_path}")
        
        # 설정 로드
        config = OptimizationConfig.from_env()
        classifier = GeminiFlashClassifier(config, prompt_text)
        
        async def run_test():
            results = await classifier.test_prompt_performance(prompt_text, samples_csv_path)
            
            print(f"\n📊 테스트 결과:")
            print(f"  - 정확도: {results.get('accuracy', 0):.4f}")
            print(f"  - 총 샘플: {results.get('total_samples', 0)}개")
            print(f"  - 정답: {results.get('correct_predictions', 0)}개")
            print(f"  - 오답: {results.get('incorrect_predictions', 0)}개")
            
            # KT 점수 계산
            accuracy = results.get('accuracy', 0)
            calculator = KTScoreCalculator()
            score_breakdown = calculator.calculate_full_score(accuracy, prompt_text)
            
            print(f"\n🏆 KT 점수: {score_breakdown.total_score:.4f}")
        
        asyncio.run(run_test())
        
    except Exception as e:
        print(f"❌ 정확도 테스트 실패: {e}")
        import traceback
        traceback.print_exc()

def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="KT 해커톤 프롬프트 최적화 시스템",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 3단계 최적화 실행
  python kt_hackathon_cli.py optimize --prompt prompt/system_prompt_v1.txt
  
  # KT 점수 계산
  python kt_hackathon_cli.py score --prompt prompt/system_prompt_v1.txt --accuracy 0.75
  
  # 샘플 데이터 분석
  python kt_hackathon_cli.py analyze-data
  
  # 프롬프트 정확도 테스트
  python kt_hackathon_cli.py test --prompt prompt/system_prompt_v1.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령어')
    
    # 최적화 명령어
    optimize_parser = subparsers.add_parser('optimize', help='3단계 최적화 실행')
    optimize_parser.add_argument('--prompt', required=True, help='초기 프롬프트 파일 경로')
    optimize_parser.add_argument('--samples', default='data/samples.csv', help='샘플 CSV 파일 경로')
    optimize_parser.add_argument('--output', default='prompt/gemini', help='출력 디렉토리')
    optimize_parser.add_argument('--target', type=float, default=0.9, help='목표 KT 점수')
    optimize_parser.add_argument('--auto', action='store_true', help='자동화된 최적화 사용 (Gemini Pro 기반)')
    
    # 점수 계산 명령어
    score_parser = subparsers.add_parser('score', help='KT 점수 계산')
    score_parser.add_argument('--prompt', required=True, help='프롬프트 파일 경로')
    score_parser.add_argument('--accuracy', type=float, required=True, help='정확도 (0.0-1.0)')
    
    # 데이터 분석 명령어
    analyze_parser = subparsers.add_parser('analyze-data', help='샘플 데이터 분석')
    analyze_parser.add_argument('--samples', default='data/samples.csv', help='샘플 CSV 파일 경로')
    
    # 테스트 명령어
    test_parser = subparsers.add_parser('test', help='프롬프트 정확도 테스트')
    test_parser.add_argument('--prompt', required=True, help='프롬프트 파일 경로')
    test_parser.add_argument('--samples', default='data/samples.csv', help='샘플 CSV 파일 경로')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # 로깅 설정
    setup_logging()
    
    # 명령어 실행
    if args.command == 'optimize':
        return asyncio.run(run_kt_optimization(
            args.prompt, args.samples, args.output, args.target, args.auto
        ))
    elif args.command == 'score':
        calculate_kt_score(args.prompt, args.accuracy)
        return 0
    elif args.command == 'analyze-data':
        analyze_samples_data(args.samples)
        return 0
    elif args.command == 'test':
        test_prompt_accuracy(args.prompt, args.samples)
        return 0
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    sys.exit(main())