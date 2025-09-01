# Design Document

## Overview

Gemini 2.5 Flash와 Gemini 2.5 Pro를 활용한 자동 프롬프트 최적화 시스템 설계입니다. samples.csv의 한국어 문장 분류 문제를 Gemini 2.5 Flash로 해결하고, 틀린 문제를 Gemini 2.5 Pro가 분석하여 프롬프트를 자동으로 개선하는 반복적 최적화 시스템을 구축합니다. 무료 API를 활용하여 비용 효율적으로 최고 성능의 프롬프트를 찾는 것이 목표입니다.

## Architecture

### 시스템 전체 구조
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CSV Reader    │───▶│  Gemini Flash    │───▶│ Result Analyzer │
│                 │    │   (Classifier)   │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Prompt Manager  │◀───│   Gemini Pro     │◀───│ Error Extractor │
│                 │    │   (Analyzer)     │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Iteration Loop  │
│   Controller    │
└─────────────────┘
```

### 핵심 컴포넌트

#### 1. GeminiFlashClassifier
- samples.csv 문제 해결
- 4가지 속성 분류 수행
- 결과 정확도 계산

#### 2. GeminiProAnalyzer  
- 틀린 문제 패턴 분석
- 프롬프트 개선점 도출
- 분석 결과 txt 저장

#### 3. PromptOptimizer
- 분석 기반 프롬프트 수정
- 버전 관리 및 백업
- 개선 로그 기록

#### 4. IterationController
- 목표 달성까지 반복 제어
- 진행 상황 모니터링
- 종료 조건 관리

## Components and Interfaces

### 1. CSV 데이터 처리 인터페이스
```python
class CSVProcessor:
    def load_samples(self, file_path: str) -> List[Sample]
    def extract_questions(self, samples: List[Sample]) -> List[str]
    def get_correct_answers(self, samples: List[Sample]) -> List[str]
```

### 2. Gemini Flash 분류 인터페이스
```python
class GeminiFlashClassifier:
    def __init__(self, api_key: str, prompt: str)
    def classify_batch(self, questions: List[str]) -> List[str]
    def classify_single(self, question: str) -> str
    def calculate_accuracy(self, predictions: List[str], answers: List[str]) -> float
```

### 3. 결과 분석 인터페이스
```python
class ResultAnalyzer:
    def compare_results(self, predictions: List[str], answers: List[str]) -> AnalysisResult
    def extract_errors(self, analysis: AnalysisResult) -> List[ErrorCase]
    def generate_error_report(self, errors: List[ErrorCase]) -> str
```

### 4. Gemini Pro 분석 인터페이스
```python
class GeminiProAnalyzer:
    def __init__(self, api_key: str)
    def analyze_errors(self, errors: List[ErrorCase], current_prompt: str) -> AnalysisReport
    def save_analysis(self, report: AnalysisReport, file_path: str)
```

### 5. 프롬프트 최적화 인터페이스
```python
class PromptOptimizer:
    def load_current_prompt(self, file_path: str) -> str
    def apply_improvements(self, prompt: str, analysis: AnalysisReport) -> str
    def save_new_version(self, prompt: str, version: int) -> str
    def log_changes(self, changes: List[str], version: int)
```

## Data Models

### Sample 데이터 모델
```python
@dataclass
class Sample:
    id: int
    sentence: str
    type: str      # 사실형/추론형/대화형/예측형
    polarity: str  # 긍정/부정/미정
    tense: str     # 과거/현재/미래
    certainty: str # 확실/불확실
```

### 오류 케이스 모델
```python
@dataclass
class ErrorCase:
    question_id: int
    sentence: str
    expected: str
    predicted: str
    error_type: str  # type/polarity/tense/certainty
```

### 분석 결과 모델
```python
@dataclass
class AnalysisReport:
    total_errors: int
    error_patterns: Dict[str, int]
    improvement_suggestions: List[str]
    prompt_modifications: List[str]
    confidence_score: float
```

### 반복 상태 모델
```python
@dataclass
class IterationState:
    iteration: int
    current_accuracy: float
    target_accuracy: float
    best_accuracy: float
    best_prompt_version: int
    is_converged: bool
```

## Error Handling

### API 오류 처리
```python
class APIErrorHandler:
    def handle_rate_limit(self, retry_count: int) -> bool
    def handle_network_error(self, error: Exception) -> bool
    def handle_response_format_error(self, response: str) -> str
    def exponential_backoff(self, attempt: int) -> float
```

### 데이터 검증
- CSV 파일 형식 검증
- 응답 형식 검증 (번호. 유형,극성,시제,확실성)
- 분류 라벨 유효성 검증
- 파일 I/O 오류 처리

### 복구 메커니즘
- API 호출 실패 시 재시도 (최대 3회)
- 부분 결과 저장 및 복구
- 프롬프트 백업 및 롤백
- 진행 상황 체크포인트

## Testing Strategy

### 단위 테스트
- 각 컴포넌트별 독립 테스트
- Mock API 응답을 통한 테스트
- 데이터 처리 로직 검증
- 오류 처리 시나리오 테스트

### 통합 테스트
- 전체 최적화 사이클 테스트
- 실제 API를 통한 E2E 테스트
- 다양한 목표 정확도 시나리오
- 장시간 실행 안정성 테스트

### 성능 테스트
- API 호출 최적화
- 배치 처리 효율성
- 메모리 사용량 모니터링
- 응답 시간 측정

## Implementation Considerations

### API 사용 최적화
- Gemini 2.5 Flash: 무료 할당량 효율적 사용
- Gemini 2.5 Pro: 분석 작업에만 사용하여 비용 절약
- 배치 처리로 API 호출 횟수 최소화
- 응답 캐싱으로 중복 호출 방지

### 프롬프트 버전 관리
```
prompt/
├── system_prompt_v1.txt
├── system_prompt_v2.txt
├── system_prompt_v3.txt
└── analysis/
    ├── analysis_v1.txt
    ├── analysis_v2.txt
    └── changes_log.txt
```

### 진행 상황 모니터링
- 실시간 정확도 추적
- 개선 추세 시각화
- 오류 패턴 분석
- 수렴 조건 모니터링

### 설정 관리
```python
@dataclass
class OptimizationConfig:
    target_accuracy: float = 0.95
    max_iterations: int = 10
    api_retry_count: int = 3
    batch_size: int = 50
    analysis_file_prefix: str = "analysis"
    prompt_file_prefix: str = "system_prompt"
```

## Performance Optimization

### 배치 처리 전략
- 문제를 배치 단위로 나누어 처리
- API 응답 시간 고려한 배치 크기 조정
- 병렬 처리 가능한 부분 식별
- 메모리 효율적인 데이터 처리

### 캐싱 전략
- 동일 문장에 대한 분류 결과 캐싱
- 분석 결과 임시 저장
- 프롬프트 변경 이력 추적
- 중간 결과 체크포인트

### 수렴 조건 최적화
- 연속 N회 개선 없음 시 조기 종료
- 목표 달성 후 추가 검증
- 최적 프롬프트 자동 선택
- 성능 저하 시 이전 버전 복구

## Security and Privacy

### API 키 관리
- 환경 변수를 통한 안전한 키 저장
- 로그에서 API 키 정보 제외
- 키 유효성 검증
- 키 순환 지원

### 데이터 보안
- 민감한 분석 데이터 로컬 저장
- 임시 파일 자동 정리
- 사용자 데이터 암호화 옵션
- 분석 결과 접근 제어