# 사용 예제

Gemini Prompt Optimizer의 다양한 사용 사례와 예제를 제공합니다.

## 목차

- [기본 사용법](#기본-사용법)
- [고급 설정](#고급-설정)
- [배치 처리](#배치-처리)
- [모니터링 및 분석](#모니터링-및-분석)
- [커스텀 워크플로우](#커스텀-워크플로우)
- [실제 사용 사례](#실제-사용-사례)

## 기본 사용법

### 1. 빠른 시작

가장 기본적인 사용법입니다.

```bash
# 1. 설정 파일 생성
python cli.py create-config

# 2. .env 파일에 API 키 설정
echo "GEMINI_API_KEY=your_api_key_here" >> .env

# 3. 샘플 프롬프트 생성
python cli.py create-prompt --output initial_prompt.txt

# 4. 최적화 실행
python cli.py optimize --initial-prompt initial_prompt.txt
```

### 2. 데이터셋 분석 먼저 하기

최적화 전에 데이터셋을 분석해보세요.

```bash
# 데이터셋 통계 확인
python cli.py analyze-dataset --csv-path data/samples.csv

# API 연결 테스트
python cli.py test-api
```

**출력 예시:**
```
📊 Dataset Statistics:
Total Samples: 1000
Valid Dataset: ✅

📊 Distribution:
Type:
  사실형: 400 (40.0%)
  추론형: 300 (30.0%)
  대화형: 200 (20.0%)
  예측형: 100 (10.0%)

Polarity:
  긍정: 600 (60.0%)
  부정: 250 (25.0%)
  미정: 150 (15.0%)
```

## 고급 설정

### 1. 커스텀 목표 설정

```bash
# 높은 정확도 목표
python cli.py optimize \
  --initial-prompt prompt.txt \
  --target-accuracy 0.98 \
  --max-iterations 20

# 빠른 테스트용 (낮은 목표)
python cli.py optimize \
  --initial-prompt prompt.txt \
  --target-accuracy 0.85 \
  --max-iterations 5
```

### 2. 배치 크기 최적화

```bash
# 대용량 데이터용 (큰 배치)
python cli.py optimize \
  --initial-prompt prompt.txt \
  --batch-size 100

# API 한도가 낮은 경우 (작은 배치)
python cli.py optimize \
  --initial-prompt prompt.txt \
  --batch-size 10
```

### 3. 출력 디렉토리 지정

```bash
# 결과를 특정 디렉토리에 저장
python cli.py optimize \
  --initial-prompt prompt.txt \
  --output-dir results/experiment_1/
```

## 배치 처리

### 1. 대용량 데이터셋 처리

```python
# large_dataset_optimizer.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.performance_optimizer import BatchProcessor

# 대용량 데이터용 설정
config = OptimizationConfig.from_env()
config.batch_size = 200  # 큰 배치 크기
config.max_iterations = 15
config.target_accuracy = 0.92

# 성능 최적화 활성화
optimizer = GeminiPromptOptimizer(config)

# 실행
result = optimizer.run_optimization("initial_prompt.txt")
print(f"처리 완료: {result.final_accuracy:.2%}")
```

### 2. 메모리 효율적 처리

```python
# memory_efficient.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.performance_optimizer import MemoryOptimizer

config = OptimizationConfig.from_env()
config.batch_size = 25  # 작은 배치로 메모리 절약

optimizer = GeminiPromptOptimizer(config)

# 메모리 사용량 모니터링
memory_info = MemoryOptimizer.get_memory_usage()
print(f"시작 메모리: {memory_info.get('rss_mb', 0):.1f} MB")

result = optimizer.run_optimization("initial_prompt.txt")

memory_info = MemoryOptimizer.get_memory_usage()
print(f"종료 메모리: {memory_info.get('rss_mb', 0):.1f} MB")
```

## 모니터링 및 분석

### 1. 실시간 모니터링

```python
# realtime_monitoring.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.monitoring import OptimizationMonitor, RealtimeMonitor

config = OptimizationConfig.from_env()
optimizer = GeminiPromptOptimizer(config)

# 모니터링 설정
monitor = OptimizationMonitor("monitoring/")
realtime = RealtimeMonitor()

monitor.start_monitoring()

# 최적화 실행 (모니터링과 함께)
result = optimizer.run_optimization("initial_prompt.txt")

# 결과 내보내기
monitor.export_metrics("monitoring/metrics.json")
print("모니터링 데이터 저장 완료")
```

### 2. 성능 분석

```python
# performance_analysis.py
from utils.performance_optimizer import PerformanceMonitor
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 성능 모니터 설정
perf_monitor = PerformanceMonitor()
perf_monitor.start_monitoring()

config = OptimizationConfig.from_env()
optimizer = GeminiPromptOptimizer(config)

# 최적화 실행
result = optimizer.run_optimization("initial_prompt.txt")

# 성능 리포트 출력
perf_monitor.print_performance_summary()

# 상세 성능 데이터
report = perf_monitor.get_performance_report()
print(f"API 호출 수: {report['api_calls']}")
print(f"평균 응답 시간: {report['average_response_time']:.2f}초")
print(f"캐시 적중률: {report['cache_hit_rate']:.1%}")
```

### 3. 시각화 리포트 생성

```python
# visualization_example.py
from utils.visualization import export_visualization_data, generate_html_report
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

config = OptimizationConfig.from_env()
optimizer = GeminiPromptOptimizer(config)

# 최적화 실행
result = optimizer.run_optimization("initial_prompt.txt")

# 시각화 데이터 준비
optimization_data = {
    'result': {
        'final_accuracy': result.final_accuracy,
        'best_accuracy': result.best_accuracy,
        'total_iterations': result.total_iterations,
        'execution_time': result.execution_time,
        'convergence_achieved': result.convergence_achieved,
        'best_prompt_version': result.best_prompt_version,
        'final_prompt_path': result.final_prompt_path
    },
    'iteration_history': optimizer.iteration_controller.export_iteration_history(),
    'target_accuracy': config.target_accuracy
}

# HTML 리포트 생성
generate_html_report(optimization_data, "results/report.html")
print("HTML 리포트 생성 완료: results/report.html")

# 모든 시각화 데이터 내보내기
export_visualization_data(optimization_data, "results/visualization/")
```

## 커스텀 워크플로우

### 1. 다중 실험 실행

```python
# multi_experiment.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
import os
from datetime import datetime

# 실험 설정들
experiments = [
    {"name": "high_accuracy", "target": 0.98, "iterations": 20},
    {"name": "balanced", "target": 0.95, "iterations": 10},
    {"name": "fast", "target": 0.90, "iterations": 5}
]

results = {}

for exp in experiments:
    print(f"\n실험 시작: {exp['name']}")
    
    # 설정
    config = OptimizationConfig.from_env()
    config.target_accuracy = exp['target']
    config.max_iterations = exp['iterations']
    config.analysis_dir = f"experiments/{exp['name']}/analysis"
    config.prompt_dir = f"experiments/{exp['name']}/prompts"
    
    # 디렉토리 생성
    config.create_directories()
    
    # 최적화 실행
    optimizer = GeminiPromptOptimizer(config)
    result = optimizer.run_optimization("initial_prompt.txt")
    
    results[exp['name']] = {
        'final_accuracy': result.final_accuracy,
        'iterations': result.total_iterations,
        'time': result.execution_time
    }
    
    print(f"완료: {result.final_accuracy:.2%} in {result.total_iterations} iterations")

# 결과 비교
print("\n=== 실험 결과 비교 ===")
for name, result in results.items():
    print(f"{name}: {result['final_accuracy']:.2%} "
          f"({result['iterations']}회, {result['time']:.1f}초)")
```

### 2. 점진적 개선

```python
# incremental_improvement.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 단계별 목표 설정
targets = [0.80, 0.85, 0.90, 0.95]
current_prompt = "initial_prompt.txt"

for i, target in enumerate(targets, 1):
    print(f"\n=== 단계 {i}: 목표 {target:.0%} ===")
    
    config = OptimizationConfig.from_env()
    config.target_accuracy = target
    config.max_iterations = 5  # 각 단계별 짧은 반복
    config.analysis_dir = f"incremental/step_{i}/analysis"
    config.prompt_dir = f"incremental/step_{i}/prompts"
    config.create_directories()
    
    optimizer = GeminiPromptOptimizer(config)
    result = optimizer.run_optimization(current_prompt)
    
    print(f"달성: {result.final_accuracy:.2%}")
    
    # 다음 단계의 시작점으로 사용
    current_prompt = result.final_prompt_path
    
    # 목표 달성 시 조기 종료
    if result.final_accuracy >= 0.95:
        print("최종 목표 달성!")
        break
```

### 3. A/B 테스트

```python
# ab_test.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 두 가지 다른 초기 프롬프트 테스트
prompts = {
    'A': 'prompt_version_a.txt',
    'B': 'prompt_version_b.txt'
}

results = {}

for version, prompt_file in prompts.items():
    print(f"\n=== 버전 {version} 테스트 ===")
    
    config = OptimizationConfig.from_env()
    config.target_accuracy = 0.95
    config.max_iterations = 10
    config.analysis_dir = f"ab_test/version_{version}/analysis"
    config.prompt_dir = f"ab_test/version_{version}/prompts"
    config.create_directories()
    
    optimizer = GeminiPromptOptimizer(config)
    result = optimizer.run_optimization(prompt_file)
    
    results[version] = result
    print(f"버전 {version}: {result.final_accuracy:.2%} "
          f"({result.total_iterations}회 반복)")

# 결과 비교
print("\n=== A/B 테스트 결과 ===")
best_version = max(results.keys(), key=lambda v: results[v].final_accuracy)
print(f"승자: 버전 {best_version}")
print(f"정확도 차이: {results[best_version].final_accuracy - results[list(results.keys())[0] if best_version == list(results.keys())[1] else list(results.keys())[1]]:.3f}")
```

## 실제 사용 사례

### 1. 프로덕션 환경 배포

```python
# production_optimizer.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
import logging
import sys
from datetime import datetime

# 프로덕션용 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'production_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

try:
    # 안정적인 설정
    config = OptimizationConfig.from_env()
    config.target_accuracy = 0.93  # 현실적인 목표
    config.max_iterations = 8      # 적당한 반복 수
    config.batch_size = 30         # 안정적인 배치 크기
    config.api_retry_count = 5     # 높은 재시도 횟수
    
    # 검증
    config.validate()
    
    logging.info("프로덕션 최적화 시작")
    
    optimizer = GeminiPromptOptimizer(config)
    result = optimizer.run_optimization("production_prompt.txt")
    
    # 결과 검증
    if result.final_accuracy >= config.target_accuracy:
        logging.info(f"성공: {result.final_accuracy:.2%} 달성")
        
        # 결과 저장
        with open("production_result.txt", "w") as f:
            f.write(result.get_final_report())
            
    else:
        logging.warning(f"목표 미달성: {result.final_accuracy:.2%}")
        
except Exception as e:
    logging.error(f"프로덕션 최적화 실패: {e}")
    sys.exit(1)
```

### 2. 연구용 상세 분석

```python
# research_analysis.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.monitoring import OptimizationMonitor
from utils.visualization import export_visualization_data
import json

# 연구용 상세 설정
config = OptimizationConfig.from_env()
config.target_accuracy = 0.97
config.max_iterations = 25
config.convergence_threshold = 0.0005  # 더 엄격한 수렴 조건
config.patience = 5

# 상세 모니터링
monitor = OptimizationMonitor("research/monitoring")
monitor.start_monitoring()

optimizer = GeminiPromptOptimizer(config)

# 최적화 실행
result = optimizer.run_optimization("research_prompt.txt")

# 상세 분석 데이터 수집
analysis_data = {
    'config': {
        'target_accuracy': config.target_accuracy,
        'max_iterations': config.max_iterations,
        'batch_size': config.batch_size,
        'convergence_threshold': config.convergence_threshold,
        'patience': config.patience
    },
    'result': {
        'final_accuracy': result.final_accuracy,
        'best_accuracy': result.best_accuracy,
        'total_iterations': result.total_iterations,
        'convergence_achieved': result.convergence_achieved,
        'execution_time': result.execution_time
    },
    'iteration_history': optimizer.iteration_controller.export_iteration_history(),
    'optimization_stats': optimizer.iteration_controller.get_optimization_statistics()
}

# 연구 데이터 저장
with open("research/analysis_data.json", "w") as f:
    json.dump(analysis_data, f, indent=2, ensure_ascii=False)

# 시각화 데이터 생성
export_visualization_data(analysis_data, "research/visualization")

# 모니터링 데이터 저장
monitor.export_metrics("research/monitoring_metrics.json")

print("연구 데이터 수집 완료")
print(f"최종 정확도: {result.final_accuracy:.4f}")
print(f"수렴 달성: {result.convergence_achieved}")
```

### 3. 자동화된 일일 최적화

```bash
#!/bin/bash
# daily_optimization.sh

# 일일 자동 최적화 스크립트
DATE=$(date +%Y%m%d)
LOG_DIR="daily_logs/$DATE"
mkdir -p "$LOG_DIR"

echo "[$DATE] 일일 최적화 시작" >> "$LOG_DIR/daily.log"

# 최적화 실행
python cli.py optimize \
  --initial-prompt prompts/daily_prompt.txt \
  --target-accuracy 0.94 \
  --max-iterations 8 \
  --output-dir "$LOG_DIR" \
  --log-level INFO >> "$LOG_DIR/optimization.log" 2>&1

# 결과 확인
if [ $? -eq 0 ]; then
    echo "[$DATE] 최적화 성공" >> "$LOG_DIR/daily.log"
    
    # 결과를 메인 프롬프트로 복사
    cp "$LOG_DIR/prompts/system_prompt_final.txt" "prompts/daily_prompt.txt"
    
    # 성공 알림 (예: 이메일, 슬랙 등)
    echo "일일 최적화 완료: $DATE" | mail -s "Optimization Success" admin@company.com
else
    echo "[$DATE] 최적화 실패" >> "$LOG_DIR/daily.log"
    
    # 실패 알림
    echo "일일 최적화 실패: $DATE" | mail -s "Optimization Failed" admin@company.com
fi
```

### 4. 다국어 지원 확장

```python
# multilingual_optimizer.py
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 언어별 설정
languages = {
    'korean': {
        'csv': 'data/korean_samples.csv',
        'prompt': 'prompts/korean_initial.txt',
        'target': 0.95
    },
    'english': {
        'csv': 'data/english_samples.csv', 
        'prompt': 'prompts/english_initial.txt',
        'target': 0.93
    }
}

results = {}

for lang, settings in languages.items():
    print(f"\n=== {lang.upper()} 최적화 ===")
    
    config = OptimizationConfig.from_env()
    config.samples_csv_path = settings['csv']
    config.target_accuracy = settings['target']
    config.analysis_dir = f"multilingual/{lang}/analysis"
    config.prompt_dir = f"multilingual/{lang}/prompts"
    config.create_directories()
    
    optimizer = GeminiPromptOptimizer(config)
    result = optimizer.run_optimization(settings['prompt'])
    
    results[lang] = result
    print(f"{lang}: {result.final_accuracy:.2%}")

# 언어별 결과 비교
print("\n=== 다국어 최적화 결과 ===")
for lang, result in results.items():
    print(f"{lang}: {result.final_accuracy:.2%} "
          f"({result.total_iterations}회 반복)")
```

이러한 예제들을 참고하여 자신의 사용 사례에 맞게 Gemini Prompt Optimizer를 활용해보세요. 각 예제는 독립적으로 실행 가능하며, 필요에 따라 조합하여 사용할 수 있습니다.