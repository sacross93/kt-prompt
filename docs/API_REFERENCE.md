# API Reference

Gemini Prompt Optimizer의 주요 클래스와 메서드에 대한 상세 문서입니다.

## 목차

- [Configuration](#configuration)
- [Main Optimizer](#main-optimizer)
- [Services](#services)
- [Models](#models)
- [Utilities](#utilities)

## Configuration

### OptimizationConfig

최적화 설정을 관리하는 클래스입니다.

```python
from config import OptimizationConfig

config = OptimizationConfig(
    gemini_api_key="your_api_key",
    target_accuracy=0.95,
    max_iterations=10,
    batch_size=50
)
```

#### 매개변수

| 매개변수 | 타입 | 기본값 | 설명 |
|----------|------|--------|------|
| `gemini_api_key` | str | 필수 | Gemini API 키 |
| `target_accuracy` | float | 0.95 | 목표 정확도 (0.0-1.0) |
| `max_iterations` | int | 10 | 최대 반복 횟수 |
| `batch_size` | int | 50 | 배치 크기 |
| `api_retry_count` | int | 3 | API 재시도 횟수 |
| `samples_csv_path` | str | "data/samples.csv" | 샘플 데이터 경로 |
| `prompt_dir` | str | "prompt" | 프롬프트 저장 디렉토리 |
| `analysis_dir` | str | "analysis" | 분석 결과 저장 디렉토리 |
| `flash_model` | str | "gemini-1.5-flash" | Flash 모델명 |
| `pro_model` | str | "gemini-1.5-pro" | Pro 모델명 |
| `convergence_threshold` | float | 0.001 | 수렴 임계값 |
| `patience` | int | 3 | 조기 종료 인내심 |

#### 메서드

##### `from_env() -> OptimizationConfig`

환경 변수에서 설정을 로드합니다.

```python
config = OptimizationConfig.from_env()
```

##### `validate() -> None`

설정 유효성을 검증합니다.

```python
config.validate()  # ValueError 발생 가능
```

##### `create_directories() -> None`

필요한 디렉토리를 생성합니다.

```python
config.create_directories()
```

## Main Optimizer

### GeminiPromptOptimizer

메인 최적화 클래스입니다.

```python
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

config = OptimizationConfig.from_env()
optimizer = GeminiPromptOptimizer(config)
```

#### 메서드

##### `run_optimization(initial_prompt_path: str) -> OptimizationResult`

전체 최적화 과정을 실행합니다.

```python
result = optimizer.run_optimization("initial_prompt.txt")
print(f"최종 정확도: {result.final_accuracy:.2%}")
```

**매개변수:**
- `initial_prompt_path`: 초기 프롬프트 파일 경로

**반환값:**
- `OptimizationResult`: 최적화 결과 객체

##### `get_optimization_summary() -> Dict[str, Any]`

최적화 요약 정보를 반환합니다.

```python
summary = optimizer.get_optimization_summary()
```

##### `export_results(output_dir: str = None) -> str`

결과를 파일로 내보냅니다.

```python
export_path = optimizer.export_results("results/")
```

## Services

### CSVProcessor

CSV 데이터 처리 서비스입니다.

```python
from services.csv_processor import CSVProcessor

processor = CSVProcessor("data/samples.csv")
samples = processor.load_samples()
```

#### 메서드

##### `load_samples() -> List[Sample]`

CSV에서 샘플 데이터를 로드합니다.

##### `extract_questions(samples: List[Sample] = None) -> List[str]`

질문 문장을 추출합니다.

##### `get_correct_answers(samples: List[Sample] = None) -> List[str]`

정답을 추출합니다.

##### `get_statistics() -> Dict[str, Any]`

데이터셋 통계를 반환합니다.

### GeminiFlashClassifier

Gemini Flash를 이용한 분류기입니다.

```python
from services.gemini_flash_classifier import GeminiFlashClassifier

classifier = GeminiFlashClassifier(config, system_prompt)
```

#### 메서드

##### `classify_single(question: str) -> str`

단일 질문을 분류합니다.

```python
result = classifier.classify_single("1. 오늘은 날씨가 좋다.")
# "1. 사실형,긍정,현재,확실"
```

##### `classify_batch(questions: List[str]) -> List[str]`

배치 질문을 분류합니다.

```python
questions = ["1. 문장1", "2. 문장2"]
results = classifier.classify_batch(questions)
```

##### `calculate_accuracy(predictions: List[str], correct_answers: List[str]) -> float`

정확도를 계산합니다.

```python
accuracy = classifier.calculate_accuracy(predictions, answers)
```

### GeminiProAnalyzer

Gemini Pro를 이용한 오류 분석기입니다.

```python
from services.gemini_pro_analyzer import GeminiProAnalyzer

analyzer = GeminiProAnalyzer(config)
```

#### 메서드

##### `analyze_errors(errors: List[ErrorCase], current_prompt: str) -> AnalysisReport`

오류를 분석하고 개선 제안을 생성합니다.

```python
report = analyzer.analyze_errors(error_cases, current_prompt)
```

##### `save_analysis(report: AnalysisReport, file_path: str = None) -> str`

분석 결과를 파일로 저장합니다.

### PromptOptimizer

프롬프트 최적화 서비스입니다.

```python
from services.prompt_optimizer import PromptOptimizer

optimizer = PromptOptimizer(config)
```

#### 메서드

##### `apply_improvements(current_prompt: str, analysis_report: AnalysisReport) -> str`

분석 결과를 바탕으로 프롬프트를 개선합니다.

```python
improved_prompt = optimizer.apply_improvements(prompt, analysis)
```

##### `save_new_version(prompt: str, base_file_path: str = None, version: int = None) -> str`

새 버전의 프롬프트를 저장합니다.

##### `calculate_korean_ratio(text: str) -> float`

한글 문자 비율을 계산합니다.

### IterationController

반복 과정을 제어하는 서비스입니다.

```python
from services.iteration_controller import IterationController

controller = IterationController(config)
```

#### 메서드

##### `initialize_optimization(total_samples: int) -> IterationState`

최적화를 초기화합니다.

##### `start_iteration(iteration: int) -> None`

새 반복을 시작합니다.

##### `update_results(accuracy: float, correct_count: int, error_count: int) -> None`

반복 결과를 업데이트합니다.

##### `check_convergence() -> bool`

수렴 여부를 확인합니다.

##### `finalize_optimization(final_prompt_path: str = None) -> OptimizationResult`

최적화를 완료하고 결과를 반환합니다.

## Models

### Sample

샘플 데이터 모델입니다.

```python
from models.data_models import Sample

sample = Sample(
    id=1,
    sentence="오늘은 날씨가 좋다.",
    type="사실형",
    polarity="긍정", 
    tense="현재",
    certainty="확실"
)
```

#### 속성

- `id`: 샘플 ID
- `sentence`: 문장
- `type`: 유형 (사실형/추론형/대화형/예측형)
- `polarity`: 극성 (긍정/부정/미정)
- `tense`: 시제 (과거/현재/미래)
- `certainty`: 확실성 (확실/불확실)

#### 메서드

##### `get_expected_output() -> str`

예상 출력 형식을 반환합니다.

```python
output = sample.get_expected_output()
# "사실형,긍정,현재,확실"
```

### ErrorCase

오류 케이스 모델입니다.

```python
from models.data_models import ErrorCase

error = ErrorCase(
    question_id=1,
    sentence="문장",
    expected="사실형,긍정,현재,확실",
    predicted="추론형,긍정,현재,확실",
    error_type="type"
)
```

### AnalysisReport

분석 리포트 모델입니다.

```python
from models.data_models import AnalysisReport

report = AnalysisReport(
    total_errors=5,
    error_patterns={"type": 3, "polarity": 2},
    improvement_suggestions=["제안1", "제안2"],
    prompt_modifications=["수정1", "수정2"],
    confidence_score=0.8,
    analysis_text="분석 내용"
)
```

### OptimizationResult

최적화 결과 모델입니다.

```python
result = OptimizationResult(
    final_accuracy=0.95,
    best_accuracy=0.96,
    best_prompt_version=3,
    total_iterations=5,
    convergence_achieved=True,
    final_prompt_path="prompt/final.txt",
    execution_time=120.5
)
```

#### 메서드

##### `get_final_report() -> str`

최종 리포트 문자열을 반환합니다.

## Utilities

### 로깅

```python
from utils.logging_utils import setup_logging

logger = setup_logging(
    log_level="INFO",
    log_file="optimization.log",
    console_output=True
)
```

### 모니터링

```python
from utils.monitoring import OptimizationMonitor

monitor = OptimizationMonitor("output_dir")
monitor.start_monitoring()
monitor.record_iteration_metrics(1, iteration_state)
```

### 성능 최적화

```python
from utils.performance_optimizer import BatchProcessor, ResponseCache

# 배치 처리
processor = BatchProcessor(batch_size=50)
results = processor.process_batches(items, process_func)

# 응답 캐싱
cache = ResponseCache()
cached_response = cache.get(prompt)
```

### 시각화

```python
from utils.visualization import generate_html_report, export_visualization_data

# HTML 리포트 생성
generate_html_report(optimization_data, "report.html")

# 시각화 데이터 내보내기
export_visualization_data(optimization_data, "viz_output/")
```

## 예외 처리

### 주요 예외 클래스

```python
from models.exceptions import (
    GeminiOptimizerError,
    APIError,
    ValidationError,
    ConfigurationError,
    FileProcessingError,
    PromptOptimizationError,
    ConvergenceError
)

try:
    result = optimizer.run_optimization("prompt.txt")
except APIError as e:
    print(f"API 오류: {e}")
except ValidationError as e:
    print(f"검증 오류: {e}")
except GeminiOptimizerError as e:
    print(f"최적화 오류: {e}")
```

## 사용 예제

### 기본 사용법

```python
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 설정 로드
config = OptimizationConfig.from_env()
config.target_accuracy = 0.95
config.max_iterations = 10

# 최적화 실행
optimizer = GeminiPromptOptimizer(config)
result = optimizer.run_optimization("initial_prompt.txt")

# 결과 출력
print(result.get_final_report())
```

### 고급 사용법

```python
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.monitoring import OptimizationMonitor

# 설정 및 모니터링
config = OptimizationConfig.from_env()
monitor = OptimizationMonitor("monitoring/")

# 최적화 실행
optimizer = GeminiPromptOptimizer(config)
monitor.start_monitoring()

result = optimizer.run_optimization("initial_prompt.txt")

# 결과 내보내기
export_dir = optimizer.export_results()
monitor.export_metrics()

print(f"결과 저장됨: {export_dir}")
```

### 커스텀 설정

```python
config = OptimizationConfig(
    gemini_api_key="your_key",
    target_accuracy=0.98,
    max_iterations=15,
    batch_size=25,
    samples_csv_path="custom_data.csv",
    prompt_dir="custom_prompts/",
    analysis_dir="custom_analysis/",
    convergence_threshold=0.005,
    patience=5
)

config.validate()
config.create_directories()

optimizer = GeminiPromptOptimizer(config)
result = optimizer.run_optimization("custom_prompt.txt")
```