# Design Document

## Overview

기존 gemini-prompt-optimizer에서 달성한 0.7점을 넘어서는 최고 성능의 한국어 문장 분류 시스템 설계입니다. 기존 시스템들의 성공 요인을 분석하고 고급 프롬프트 엔지니어링 기법을 적용하여 Gemini 2.5 Flash에서 0.7점 이상, GPT-4o에서 최고 성능을 달성하는 자동화된 최적화 파이프라인을 구축합니다.

## Architecture

### 전체 시스템 구조
```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Prompt Analyzer    │───▶│  Advanced Prompt     │───▶│  Gemini Flash       │
│  (기존 분석)        │    │  Generator           │    │  Tester             │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
                                      │                           │
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Strategy Manager   │◀───│  Gemini Pro          │◀───│  Performance        │
│  (전략 관리)        │    │  Analyzer            │    │  Analyzer           │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
         │                           │                           │
         ▼                           ▼                           ▼
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│  Iteration          │    │  GPT-4o Final        │    │  Result Manager     │
│  Controller         │    │  Validator           │    │  (결과 관리)        │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### 핵심 컴포넌트

#### 1. PromptAnalyzer (프롬프트 분석기)
- 기존 프롬프트들의 성능 분석
- 0.7점 달성 프롬프트의 성공 요인 추출
- 개선 가능 영역 식별

#### 2. AdvancedPromptGenerator (고급 프롬프트 생성기)
- Few-shot learning 예시 동적 선택
- Chain-of-Thought 추론 과정 포함
- 다양한 프롬프트 전략 적용

#### 3. GeminiFlashTester (Gemini Flash 테스터)
- 전체 samples.csv 테스트 수행
- 정확도 계산 및 0.7점 이상 여부 판단
- 상세한 오류 분석 제공

#### 4. GeminiProAnalyzer (Gemini Pro 분석기)
- 실패 원인 심층 진단
- 속성별 문제점 구체적 식별
- 고급 개선 방안 도출

#### 5. StrategyManager (전략 관리자)
- 다양한 최적화 전략 관리
- 전략별 효과 정량적 비교
- 최적 전략 조합 탐색

#### 6. GPT4oFinalValidator (GPT-4o 최종 검증기)
- 0.7점 이상 달성 시 GPT-4o 테스트
- 기존 최고 성능과 비교
- 최종 성능 개선 확인

## Components and Interfaces

### 1. 프롬프트 분석 인터페이스
```python
class PromptAnalyzer:
    def analyze_existing_prompts(self, prompt_dir: str) -> AnalysisResult
    def identify_success_factors(self, best_prompt: str, score: float) -> List[str]
    def extract_improvement_areas(self, analysis: AnalysisResult) -> List[str]
    def generate_baseline_insights(self) -> BaselineInsights
```

### 2. 고급 프롬프트 생성 인터페이스
```python
class AdvancedPromptGenerator:
    def generate_few_shot_examples(self, error_patterns: List[str]) -> List[str]
    def add_chain_of_thought(self, base_prompt: str) -> str
    def apply_strategy(self, strategy: PromptStrategy, base_prompt: str) -> str
    def create_enhanced_prompt(self, insights: BaselineInsights) -> str
```

### 3. 성능 테스트 인터페이스
```python
class GeminiFlashTester:
    def test_full_dataset(self, prompt: str) -> TestResult
    def calculate_accuracy(self, predictions: List[str], answers: List[str]) -> float
    def analyze_errors(self, test_result: TestResult) -> ErrorAnalysis
    def check_target_achievement(self, accuracy: float, target: float = 0.7) -> bool
```

### 4. 고급 분석 인터페이스
```python
class GeminiProAnalyzer:
    def diagnose_failure_causes(self, error_analysis: ErrorAnalysis) -> DiagnosisReport
    def identify_attribute_issues(self, errors: List[ErrorCase]) -> AttributeIssues
    def suggest_advanced_improvements(self, diagnosis: DiagnosisReport) -> ImprovementPlan
    def generate_next_iteration_guide(self, plan: ImprovementPlan) -> IterationGuide
```

### 5. 전략 관리 인터페이스
```python
class StrategyManager:
    def get_available_strategies(self) -> List[PromptStrategy]
    def select_next_strategy(self, history: OptimizationHistory) -> PromptStrategy
    def evaluate_strategy_effectiveness(self, strategy: PromptStrategy, result: TestResult) -> float
    def find_optimal_combination(self, strategies: List[PromptStrategy]) -> StrategyCombo
```

## Data Models

### 기존 프롬프트 분석 결과
```python
@dataclass
class AnalysisResult:
    prompt_performances: Dict[str, float]  # 프롬프트별 성능
    success_factors: List[str]             # 성공 요인들
    common_patterns: List[str]             # 공통 패턴들
    improvement_areas: List[str]           # 개선 영역들
    best_prompt_features: List[str]        # 최고 성능 프롬프트 특징
```

### 고급 오류 분석 모델
```python
@dataclass
class ErrorAnalysis:
    total_errors: int
    attribute_errors: Dict[str, int]       # 속성별 오류 수
    error_patterns: Dict[str, List[str]]   # 패턴별 오류 사례
    boundary_cases: List[ErrorCase]        # 경계 사례들
    parsing_failures: List[str]            # 파싱 실패 사례
    confidence_scores: Dict[str, float]    # 속성별 신뢰도
```

### 진단 리포트 모델
```python
@dataclass
class DiagnosisReport:
    root_causes: List[str]                 # 근본 원인들
    attribute_issues: AttributeIssues      # 속성별 문제점
    prompt_weaknesses: List[str]           # 프롬프트 약점들
    recommended_fixes: List[str]           # 권장 수정사항
    priority_areas: List[str]              # 우선순위 영역
```

### 프롬프트 전략 모델
```python
@dataclass
class PromptStrategy:
    name: str
    description: str
    technique: str                         # few-shot, CoT, explicit-rules 등
    parameters: Dict[str, Any]
    expected_improvement: List[str]        # 예상 개선 영역
    compatibility: List[str]               # 호환 가능한 다른 전략들
```

### 최적화 히스토리 모델
```python
@dataclass
class OptimizationHistory:
    iterations: List[IterationResult]
    best_score: float
    best_prompt: str
    strategy_effectiveness: Dict[str, float]
    convergence_status: str
    total_improvements: int
```

## Error Handling

### 고급 파싱 오류 처리
```python
class AdvancedParsingHandler:
    def try_multiple_parsing_strategies(self, response: str) -> Optional[List[str]]
    def normalize_response_format(self, response: str) -> str
    def extract_partial_results(self, response: str) -> PartialResult
    def generate_parsing_feedback(self, failures: List[str]) -> str
```

### API 호출 최적화
- Gemini 2.5 Flash: 배치 처리로 효율성 극대화
- Gemini 2.5 Pro: 분석 작업에만 선택적 사용
- GPT-4o: 최종 검증 단계에서만 사용
- 지수 백오프와 재시도 로직 강화

### 시스템 복구 메커니즘
- 진행 상황 자동 저장 (체크포인트)
- 실패 지점에서 재시작 가능
- 부분 결과 보존 및 활용
- 오류 상황별 대응 전략

## Testing Strategy

### 성능 벤치마킹
```python
class PerformanceBenchmark:
    def compare_with_baseline(self, new_score: float, baseline: float = 0.7) -> Comparison
    def measure_improvement_rate(self, history: OptimizationHistory) -> float
    def evaluate_consistency(self, multiple_runs: List[float]) -> ConsistencyReport
    def generate_performance_report(self, results: List[TestResult]) -> PerformanceReport
```

### A/B 테스트 프레임워크
- 다양한 프롬프트 전략 동시 테스트
- 통계적 유의성 검증
- 성능 변화 추이 분석
- 최적 조합 탐색

### 견고성 테스트
- 다양한 문장 유형에 대한 안정성
- 극단적 케이스 처리 능력
- 장시간 실행 안정성
- 메모리 사용량 최적화

## Implementation Considerations

### 프롬프트 엔지니어링 기법

#### 1. Few-shot Learning 최적화
```python
def select_optimal_examples(error_patterns: List[str], sample_pool: List[Sample]) -> List[Sample]:
    """오류 패턴을 기반으로 최적의 few-shot 예시 선택"""
    # 오류가 많이 발생하는 패턴의 정답 예시 우선 선택
    # 경계 사례를 포함한 대표적 예시 선택
    # 4가지 속성 모두를 다루는 균형잡힌 예시 구성
```

#### 2. Chain-of-Thought 추론
```python
def add_reasoning_process(base_prompt: str) -> str:
    """단계별 추론 과정을 포함한 프롬프트 생성"""
    # 1단계: 문장 구조 분석
    # 2단계: 각 속성별 판단 근거
    # 3단계: 최종 분류 결정
    # 4단계: 형식화된 출력
```

#### 3. 명시적 규칙 강화
```python
def enhance_classification_rules(insights: BaselineInsights) -> str:
    """분석 결과를 바탕으로 분류 규칙 강화"""
    # 성공 요인을 명시적 규칙으로 변환
    # 경계 사례 처리 규칙 추가
    # 예외 상황 대응 지침 포함
```

### 디렉토리 구조
```
prompt/
├── gemini/
│   ├── baseline_v1.txt
│   ├── enhanced_v2.txt
│   ├── few_shot_v3.txt
│   ├── cot_v4.txt
│   └── final_optimized.txt
└── analysis/
    ├── performance_history.json
    ├── strategy_effectiveness.json
    └── optimization_log.txt

gemini/
├── prompt_analyzer.py
├── advanced_generator.py
├── gemini_tester.py
├── performance_tracker.py
└── main_optimizer.py
```

### 최적화 전략 우선순위

#### Phase 1: 기본 개선 (목표: 0.75점)
1. 기존 최고 성능 프롬프트 분석 및 개선
2. 명시적 분류 규칙 강화
3. 출력 형식 지침 최적화

#### Phase 2: 고급 기법 적용 (목표: 0.8점)
1. Few-shot learning 예시 추가
2. Chain-of-Thought 추론 과정 포함
3. 경계 사례 처리 규칙 세분화

#### Phase 3: 전략 조합 최적화 (목표: 0.85점+)
1. 다양한 전략 조합 실험
2. 하이퍼파라미터 튜닝
3. 모델별 최적화 (Gemini vs GPT-4o)

## Performance Optimization

### 배치 처리 최적화
- samples.csv를 적절한 크기로 분할
- 병렬 처리 가능한 부분 식별
- API 호출 횟수 최소화
- 응답 캐싱으로 중복 호출 방지

### 메모리 효율성
- 대용량 데이터 스트리밍 처리
- 중간 결과 압축 저장
- 불필요한 데이터 즉시 해제
- 메모리 사용량 실시간 모니터링

### 실행 시간 최적화
- 병렬 API 호출 (가능한 경우)
- 조기 종료 조건 설정
- 점진적 개선 추적
- 수렴 감지 및 자동 종료

## Security and Privacy

### API 키 관리
- 환경 변수 기반 안전한 저장
- 키별 사용량 추적
- 자동 키 순환 지원
- 로그에서 민감 정보 제외

### 데이터 보안
- 분석 결과 로컬 암호화 저장
- 임시 파일 자동 정리
- 사용자 데이터 익명화
- 접근 권한 제어

## Monitoring and Analytics

### 실시간 모니터링
```python
class OptimizationMonitor:
    def track_performance_trend(self, history: OptimizationHistory) -> Trend
    def detect_convergence(self, recent_scores: List[float]) -> bool
    def estimate_remaining_iterations(self, current_progress: float) -> int
    def generate_progress_visualization(self, data: MonitoringData) -> Chart
```

### 성능 분석 대시보드
- 실시간 성능 추이 그래프
- 속성별 개선 현황
- 전략별 효과 비교
- 예상 완료 시간

### 결과 리포팅
- 전체 최적화 과정 요약
- 최종 성능 개선 결과
- 사용된 전략과 효과
- 향후 개선 권장사항