# Requirements Document

## Introduction

기존 gemini-prompt-optimizer에서 달성한 0.7점을 넘어서는 최고 성능의 한국어 문장 분류 시스템을 개발하는 프로젝트입니다. Gemini 2.5 Flash로 테스트하여 0.7점 이상 달성 후 GPT-4o로 최종 검증하는 고급 프롬프트 최적화 시스템을 구축합니다. Gemini 2.5 Pro가 실패 원인을 진단하고 지속적으로 개선하는 자동화된 최적화 파이프라인을 통해 최고 성능을 달성하는 것이 목표입니다.

## Requirements

### Requirement 1

**User Story:** 개발자로서 기존 프롬프트들의 성능을 분석하고 0.7점을 넘어서는 개선된 프롬프트를 생성하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 기존 프롬프트들이 제공되면 THEN 각각의 성능과 특징을 분석해야 합니다
2. WHEN 성능 분석이 완료되면 THEN 0.7점을 달성한 프롬프트의 성공 요인을 식별해야 합니다
3. WHEN 성공 요인이 파악되면 THEN 이를 바탕으로 개선된 프롬프트를 생성해야 합니다
4. WHEN 새 프롬프트가 생성되면 THEN prompt/gemini/ 폴더에 버전 관리하여 저장해야 합니다

### Requirement 2

**User Story:** 개발자로서 Gemini 2.5 Flash를 이용해 프롬프트 성능을 정확하게 측정하고 0.7점 이상 달성 여부를 판단하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 새로운 프롬프트가 생성되면 THEN Gemini 2.5 Flash로 전체 samples.csv를 테스트해야 합니다
2. WHEN 테스트가 완료되면 THEN 정확도를 계산하고 0.7점 이상인지 확인해야 합니다
3. WHEN 0.7점 미만이면 THEN 실패 원인을 상세히 분석해야 합니다
4. WHEN 0.7점 이상이면 THEN GPT-4o로 최종 검증을 진행해야 합니다

### Requirement 3

**User Story:** 개발자로서 Gemini 2.5 Pro를 이용해 실패 원인을 진단하고 구체적인 개선 방안을 도출하는 고급 분석 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 0.7점 미만의 결과가 나오면 THEN Gemini 2.5 Pro가 오류 패턴을 심층 분석해야 합니다
2. WHEN 분석이 완료되면 THEN 각 분류 속성별 문제점을 구체적으로 식별해야 합니다
3. WHEN 문제점이 파악되면 THEN 프롬프트 엔지니어링 기법을 적용한 개선안을 제시해야 합니다
4. WHEN 개선안이 도출되면 THEN 다음 반복을 위한 구체적인 수정 지침을 제공해야 합니다

### Requirement 4

**User Story:** 개발자로서 고급 프롬프트 엔지니어링 기법을 적용하여 분류 성능을 극대화하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 프롬프트를 개선할 때 THEN Few-shot learning 예시를 동적으로 선택하여 포함해야 합니다
2. WHEN 복잡한 분류 문제가 있을 때 THEN Chain-of-Thought 추론 과정을 포함해야 합니다
3. WHEN 경계 사례가 많을 때 THEN 명확한 분류 기준과 예외 처리 규칙을 추가해야 합니다
4. WHEN 파싱 오류가 발생할 때 THEN 출력 형식 지침을 더욱 강화해야 합니다

### Requirement 5

**User Story:** 개발자로서 0.7점 이상 달성 시 GPT-4o로 최종 검증하여 실제 성능 개선을 확인하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN Gemini 2.5 Flash에서 0.7점 이상 달성하면 THEN 자동으로 GPT-4o 테스트를 시작해야 합니다
2. WHEN GPT-4o 테스트가 완료되면 THEN 기존 최고 성능과 비교해야 합니다
3. WHEN 성능이 개선되었으면 THEN 최종 프롬프트로 저장하고 결과를 문서화해야 합니다
4. WHEN 성능이 개선되지 않았으면 THEN 추가 최적화를 위한 분석을 수행해야 합니다

### Requirement 6

**User Story:** 개발자로서 자동화된 반복 최적화 과정을 통해 목표 성능에 도달할 때까지 지속적으로 개선하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 목표 성능에 도달하지 못했으면 THEN 자동으로 다음 최적화 사이클을 시작해야 합니다
2. WHEN 각 반복에서 THEN 이전 반복의 학습 내용을 누적하여 활용해야 합니다
3. WHEN 연속 N회 개선이 없으면 THEN 다른 최적화 전략을 시도해야 합니다
4. WHEN 최대 반복 횟수에 도달하면 THEN 최고 성능 프롬프트를 최종 결과로 제시해야 합니다

### Requirement 7

**User Story:** 개발자로서 프롬프트와 테스트 코드를 체계적으로 관리하고 실험 결과를 추적할 수 있는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 새로운 프롬프트가 생성되면 THEN prompt/gemini/ 폴더에 버전별로 저장해야 합니다
2. WHEN 테스트 코드를 작성할 때 THEN gemini/ 폴더에 Python 스크립트로 저장해야 합니다
3. WHEN 실험이 완료되면 THEN 성능 지표와 개선 내용을 상세히 기록해야 합니다
4. WHEN 최종 결과가 나오면 THEN 전체 최적화 과정과 결과를 종합 리포트로 작성해야 합니다

### Requirement 8

**User Story:** 개발자로서 파싱 오류와 API 호출 실패를 최소화하고 안정적인 성능 측정을 보장하는 견고한 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 파싱 오류가 발생하면 THEN 다양한 파싱 전략을 시도하여 복구해야 합니다
2. WHEN API 호출이 실패하면 THEN 지수 백오프와 재시도 로직으로 복구해야 합니다
3. WHEN 응답 형식이 예상과 다르면 THEN 자동으로 형식을 정규화해야 합니다
4. WHEN 시스템 오류가 발생하면 THEN 현재 진행 상황을 저장하고 복구 가능하도록 해야 합니다

### Requirement 9

**User Story:** 개발자로서 실시간 성능 모니터링과 최적화 진행 상황을 시각적으로 추적할 수 있는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 각 반복이 시작되면 THEN 현재 목표와 진행 상황을 명확히 표시해야 합니다
2. WHEN 테스트가 완료되면 THEN 성능 변화 추이를 그래프로 시각화해야 합니다
3. WHEN 분석이 완료되면 THEN 주요 개선점과 다음 단계를 요약해서 출력해야 합니다
4. WHEN 최적화가 완료되면 THEN 전체 과정의 성능 개선 히스토리를 제공해야 합니다

### Requirement 10

**User Story:** 개발자로서 다양한 프롬프트 전략을 실험하고 최적의 조합을 찾는 고급 최적화 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 기본 최적화가 정체되면 THEN 다른 프롬프트 전략을 자동으로 시도해야 합니다
2. WHEN 여러 전략을 시도할 때 THEN 각각의 효과를 정량적으로 비교해야 합니다
3. WHEN 최적 전략이 식별되면 THEN 해당 전략을 중심으로 세밀 조정을 수행해야 합니다
4. WHEN 모든 전략을 시도했으면 THEN 최고 성능 조합을 최종 결과로 제시해야 합니다