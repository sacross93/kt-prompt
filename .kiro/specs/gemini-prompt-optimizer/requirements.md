# Requirements Document

## Introduction

Gemini 2.5 Flash를 활용한 자동 프롬프트 최적화 시스템 개발 프로젝트입니다. samples.csv의 한국어 문장 분류 문제를 Gemini 2.5 Flash로 풀고, 틀린 문제를 Gemini 2.5 Pro가 분석하여 프롬프트를 자동으로 개선하는 반복적 최적화 시스템을 구축하는 것이 목표입니다.

## Requirements

### Requirement 1

**User Story:** 개발자로서 Gemini 2.5 Flash를 이용해 samples.csv의 문제를 자동으로 풀고 결과를 분석할 수 있는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN samples.csv 파일이 제공되면 THEN Gemini 2.5 Flash가 각 문장을 4가지 속성으로 분류해야 합니다
2. WHEN 분류 결과가 나오면 THEN 정답과 비교하여 정확도를 계산해야 합니다
3. WHEN 틀린 문제가 있으면 THEN 틀린 문제 번호, 입력 문장, 예상 정답, 실제 출력을 추출해야 합니다
4. WHEN 분류 작업이 완료되면 THEN 전체 정확도와 틀린 문제 리스트를 반환해야 합니다

### Requirement 2

**User Story:** 개발자로서 Gemini 2.5 Pro를 이용해 틀린 문제와 현재 프롬프트를 분석하여 개선점을 도출하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 틀린 문제 리스트와 현재 프롬프트가 제공되면 THEN Gemini 2.5 Pro가 오류 패턴을 분석해야 합니다
2. WHEN 분석이 완료되면 THEN 어떤 분류 기준이 문제였는지 구체적으로 식별해야 합니다
3. WHEN 분석 결과가 나오면 THEN 프롬프트 개선 방향을 제시해야 합니다
4. WHEN 분석이 완료되면 THEN 분석 내용을 지정된 txt 파일에 저장해야 합니다

### Requirement 3

**User Story:** 개발자로서 분석 결과를 바탕으로 프롬프트를 자동으로 수정하고 개선된 프롬프트를 생성하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 분석 결과가 제공되면 THEN 현재 프롬프트의 문제점을 식별해야 합니다
2. WHEN 개선점이 도출되면 THEN 기존 프롬프트를 수정하여 새로운 버전을 생성해야 합니다
3. WHEN 새 프롬프트가 생성되면 THEN 버전 관리를 위해 파일명에 버전 정보를 포함해야 합니다
4. WHEN 프롬프트 수정이 완료되면 THEN 수정된 내용과 이유를 로그로 기록해야 합니다

### Requirement 4

**User Story:** 개발자로서 목표 정확도에 도달할 때까지 테스트-분석-개선 과정을 자동으로 반복하는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 목표 정확도가 파라미터로 설정되면 THEN 해당 목표에 도달할 때까지 반복해야 합니다
2. WHEN 각 반복에서 THEN 현재 정확도가 목표에 도달했는지 확인해야 합니다
3. WHEN 목표에 도달하지 못했으면 THEN 자동으로 다음 최적화 사이클을 시작해야 합니다
4. WHEN 목표에 도달하거나 최대 반복 횟수에 도달하면 THEN 최적화 과정을 종료해야 합니다

### Requirement 5

**User Story:** 개발자로서 최적화 과정의 진행 상황과 결과를 추적하고 모니터링할 수 있는 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN 각 반복이 시작되면 THEN 현재 반복 횟수와 목표를 출력해야 합니다
2. WHEN 테스트가 완료되면 THEN 현재 정확도와 이전 대비 개선 정도를 표시해야 합니다
3. WHEN 분석이 완료되면 THEN 주요 개선점과 수정 내용을 요약해서 출력해야 합니다
4. WHEN 최적화가 완료되면 THEN 최종 결과와 최적 프롬프트 정보를 제공해야 합니다

### Requirement 6

**User Story:** 개발자로서 Gemini API를 안정적으로 사용하고 오류 상황에 대응할 수 있는 견고한 시스템을 만들고 싶습니다.

#### Acceptance Criteria

1. WHEN API 호출이 실패하면 THEN 재시도 로직을 통해 복구를 시도해야 합니다
2. WHEN 응답 형식이 예상과 다르면 THEN 오류를 감지하고 적절히 처리해야 합니다
3. WHEN 네트워크 오류가 발생하면 THEN 사용자에게 명확한 오류 메시지를 제공해야 합니다
4. WHEN 무료 할당량이 초과되면 THEN 적절한 대기 시간을 두고 재시도해야 합니다

### Requirement 7

**User Story:** 개발자로서 파싱 에러를 해결하고 더 높은 정확도를 달성하기 위해 시스템을 개선하고 싶습니다.

#### Acceptance Criteria

1. WHEN 분류 결과 파싱이 실패하면 THEN 구체적인 오류 원인을 로깅하고 복구를 시도해야 합니다
2. WHEN "string indices must be integers" 오류가 발생하면 THEN 데이터 구조 접근 방식을 수정해야 합니다
3. WHEN 파싱 실패율이 높으면 THEN 출력 형식 지침을 더 명확하게 개선해야 합니다
4. WHEN 현재 최고 성능(0.7점)에서 THEN 추가 최적화를 통해 더 높은 정확도를 달성해야 합니다

### Requirement 8

**User Story:** 개발자로서 프롬프트 엔지니어링 기법을 적용하여 분류 성능을 극대화하고 싶습니다.

#### Acceptance Criteria

1. WHEN 현재 프롬프트를 분석하면 THEN 개선 가능한 부분을 식별해야 합니다
2. WHEN Few-shot 예시를 추가하면 THEN 분류 정확도가 향상되어야 합니다
3. WHEN 분류 기준을 더 명확하게 정의하면 THEN 일관성이 개선되어야 합니다
4. WHEN Chain-of-Thought 기법을 적용하면 THEN 복잡한 문장의 분류 정확도가 향상되어야 합니다