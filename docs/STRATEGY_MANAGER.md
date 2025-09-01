# StrategyManager 시스템 문서

## 개요

StrategyManager는 프롬프트 최적화 전략을 체계적으로 관리하고 효과를 정량적으로 비교하는 고급 시스템입니다. 다양한 프롬프트 엔지니어링 기법의 성능을 추적하고, 최적의 전략 조합을 자동으로 탐색하며, 지속적인 학습을 통해 최고 성능을 달성합니다.

## 주요 기능

### 1. 전략 관리 (Strategy Management)
- **다양한 프롬프트 전략 지원**: Baseline, Few-shot Learning, Chain-of-Thought, Explicit Rules, Hybrid
- **전략 조합 관리**: 단일 전략부터 복합 전략까지 체계적 관리
- **동적 전략 선택**: 현재 상황에 맞는 최적 전략 자동 선택

### 2. 성능 평가 (Performance Evaluation)
- **종합 효과성 점수**: 정확도(70%) + 파싱 성공률(20%) + 실행 시간(10%) 가중 평균
- **상세 메트릭 추적**: 속성별 정확도, 오류 패턴, 실행 통계
- **비교 분석**: 전략별 성능 순위 및 효과 비교

### 3. 학습 및 최적화 (Learning & Optimization)
- **히스토리 추적**: 모든 실험 결과와 개선 과정 기록
- **수렴 감지**: 성능 정체 상황 자동 감지
- **적응적 전략 선택**: 상황에 따른 탐색적/활용적 전략 전환

## 시스템 구조

### 핵심 클래스

#### StrategyManager
```python
class StrategyManager:
    def __init__(self, analysis_dir: str = "prompt/analysis")
    def get_available_strategies(self) -> List[PromptStrategy]
    def record_strategy_result(self, result: StrategyResult)
    def select_next_strategy(self, history: OptimizationHistory) -> PromptStrategy
    def find_optimal_combination(self, strategies: List[PromptStrategy]) -> StrategyCombo
    def get_strategy_comparison(self) -> Dict[str, Any]
    def save_strategy_data(self)
```

#### StrategyResult
```python
@dataclass
class StrategyResult:
    strategy: PromptStrategy
    accuracy: float
    execution_time: float
    error_count: int
    parsing_failures: int
    timestamp: str
    prompt_version: str
    detailed_metrics: Dict[str, float]
```

#### StrategyCombo
```python
@dataclass
class StrategyCombo:
    strategies: List[PromptStrategy]
    combo_name: str
    description: str
    expected_synergy: float
```

### 데이터 모델

#### OptimizationHistory
- **iterations**: 모든 실험 반복 기록
- **best_score**: 최고 달성 성능
- **strategy_effectiveness**: 전략별 효과성 점수
- **convergence_status**: 수렴 상태 추적

## 사용 방법

### 1. 기본 사용법

```python
from strategy_manager import StrategyManager, StrategyResult
from advanced_generator import PromptStrategy

# 매니저 초기화
manager = StrategyManager()

# 전략 실행 결과 기록
result = StrategyResult(
    strategy=PromptStrategy.FEW_SHOT,
    accuracy=0.75,
    execution_time=30.0,
    error_count=25,
    parsing_failures=1,
    timestamp=datetime.now().isoformat(),
    prompt_version="v1",
    detailed_metrics={"type_accuracy": 0.8}
)

manager.record_strategy_result(result)

# 다음 전략 선택
next_strategy = manager.select_next_strategy(manager.optimization_history)
```

### 2. 최적화 파이프라인 통합

```python
class OptimizationPipeline:
    def __init__(self):
        self.strategy_manager = StrategyManager()
        self.prompt_generator = AdvancedPromptGenerator()
    
    def run_optimization_cycle(self):
        while not self.is_target_reached():
            # 1. 전략 선택
            strategy = self.strategy_manager.select_next_strategy(
                self.strategy_manager.optimization_history
            )
            
            # 2. 프롬프트 생성
            prompt = self.prompt_generator.create_enhanced_prompt(strategy)
            
            # 3. 테스트 실행
            result = self.run_test(prompt, strategy)
            
            # 4. 결과 기록
            self.strategy_manager.record_strategy_result(result)
```

### 3. 전략 조합 최적화

```python
# 최적 조합 탐색
strategies = [PromptStrategy.FEW_SHOT, PromptStrategy.EXPLICIT_RULES]
optimal_combo = manager.find_optimal_combination(strategies)

print(f"최적 조합: {optimal_combo.combo_name}")
print(f"예상 시너지: {optimal_combo.expected_synergy}")
```

## 전략 선택 알고리즘

### 1. 기본 전략 선택
- **최고 효과성 전략**: 현재까지 가장 높은 효과성을 보인 전략 선택
- **미시도 전략 우선**: 아직 시도하지 않은 전략이 있으면 우선 선택

### 2. 성능 정체 시 대안 전략
- **대안 전략 탐색**: 연속 N회 개선이 없으면 다른 전략 시도
- **효과성 역순 선택**: 낮은 효과성 전략부터 재시도

### 3. 수렴 시 탐색적 전략
- **조합 전략 시도**: 단일 전략 수렴 시 조합 전략 탐색
- **하이브리드 전략**: 모든 기법을 통합한 복합 전략 적용

## 성능 메트릭

### 효과성 점수 계산
```
효과성 = 정확도 × 0.7 + 파싱성공률 × 0.2 + 시간효율성 × 0.1
```

- **정확도**: 분류 정확도 (0.0 ~ 1.0)
- **파싱 성공률**: 1 - (파싱실패 / 총오류수)
- **시간 효율성**: min(1.0, 60초 / 실행시간)

### 수렴 감지
```python
def is_converged(self, threshold: float = 0.01, window: int = 3) -> bool:
    recent_scores = self.get_recent_performance(window)
    variance = calculate_variance(recent_scores)
    return variance < threshold
```

## 데이터 저장 형식

### strategy_effectiveness.json
```json
{
  "strategies": {
    "few_shot": {
      "effectiveness": 0.823,
      "used_count": 3,
      "description": "Few-shot learning 예시 추가"
    }
  },
  "optimization_history": {
    "best_score": 0.78,
    "best_strategy": "hybrid",
    "total_improvements": 4
  }
}
```

### strategy_detailed_results.json
```json
{
  "few_shot": [
    {
      "strategy": "few_shot",
      "accuracy": 0.75,
      "execution_time": 35.0,
      "detailed_metrics": {
        "type_accuracy": 0.8,
        "polarity_accuracy": 0.72
      }
    }
  ]
}
```

## 통합 예제

### 완전한 최적화 사이클
```python
def run_complete_optimization():
    manager = StrategyManager()
    generator = AdvancedPromptGenerator()
    
    target_accuracy = 0.7
    max_iterations = 10
    
    for iteration in range(max_iterations):
        # 전략 선택
        strategy = manager.select_next_strategy(manager.optimization_history)
        
        # 프롬프트 생성
        prompt = generator.create_enhanced_prompt(strategy)
        
        # 테스트 실행 (Gemini API 호출)
        test_result = run_gemini_test(prompt)
        
        # 결과 기록
        result = StrategyResult(
            strategy=strategy,
            accuracy=test_result.accuracy,
            execution_time=test_result.time,
            error_count=test_result.errors,
            parsing_failures=test_result.parsing_errors,
            timestamp=datetime.now().isoformat(),
            prompt_version=f"v{iteration}",
            detailed_metrics=test_result.detailed_metrics
        )
        
        manager.record_strategy_result(result)
        
        # 목표 달성 확인
        if test_result.accuracy >= target_accuracy:
            print("목표 성능 달성!")
            break
    
    # 최종 리포트
    report = manager.generate_strategy_report()
    print(report)
    
    # 데이터 저장
    manager.save_strategy_data()
```

## 확장 가능성

### 1. 새로운 전략 추가
```python
class CustomStrategy(Enum):
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    META_LEARNING = "meta_learning"

# StrategyManager에 새 전략 등록
manager.register_custom_strategy(CustomStrategy.REINFORCEMENT_LEARNING)
```

### 2. 커스텀 효과성 계산
```python
def custom_effectiveness_calculator(result: StrategyResult) -> float:
    # 사용자 정의 효과성 계산 로직
    return custom_score

manager.set_effectiveness_calculator(custom_effectiveness_calculator)
```

### 3. 고급 조합 전략
```python
def create_adaptive_combo(performance_history: List[float]) -> StrategyCombo:
    # 성능 히스토리 기반 적응적 조합 생성
    return adaptive_combo

manager.register_combo_generator(create_adaptive_combo)
```

## 모니터링 및 디버깅

### 실시간 모니터링
```python
# 성능 추이 시각화
comparison = manager.get_strategy_comparison()
plot_performance_trends(comparison["performance_trends"])

# 전략별 효과성 비교
plot_strategy_effectiveness(comparison["effectiveness_scores"])
```

### 디버깅 정보
```python
# 상세 분석 리포트
report = manager.generate_strategy_report()
print(report)

# 수렴 상태 확인
is_converged = manager.optimization_history.is_converged()
recent_performance = manager.optimization_history.get_recent_performance(5)
```

## 베스트 프랙티스

### 1. 전략 실험 설계
- **충분한 반복**: 각 전략당 최소 3회 이상 테스트
- **일관된 환경**: 동일한 테스트 데이터와 평가 기준 사용
- **점진적 개선**: 급진적 변화보다 점진적 개선 추구

### 2. 성능 모니터링
- **정기적 저장**: 실험 결과를 정기적으로 저장
- **백업 관리**: 중요한 실험 데이터의 백업 유지
- **버전 관리**: 프롬프트 버전과 결과를 연결하여 추적

### 3. 최적화 전략
- **조기 종료**: 목표 성능 달성 시 조기 종료
- **탐색-활용 균형**: 탐색과 활용의 적절한 균형 유지
- **메타 학습**: 전략 선택 자체도 학습하여 개선

이 문서는 StrategyManager 시스템의 완전한 가이드입니다. 추가 질문이나 사용 사례가 있으면 언제든 문의하세요.