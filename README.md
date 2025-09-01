# 🤖 KT Prompt Optimizer

Gemini 2.5 Flash와 Pro를 활용한 자동 프롬프트 최적화 시스템입니다. 한국어 문장 분류 문제를 해결하기 위해 프롬프트를 자동으로 분석하고 개선합니다.

## ✨ 주요 기능

- **자동 프롬프트 최적화**: Gemini Pro가 오류를 분석하고 프롬프트를 자동 개선
- **반복적 학습**: 목표 정확도 달성까지 자동으로 반복 최적화
- **실시간 모니터링**: 최적화 진행 상황을 실시간으로 추적
- **성능 최적화**: 배치 처리, 캐싱, 메모리 최적화로 효율적 실행
- **상세한 분석**: 오류 패턴 분석 및 개선 제안
- **시각화 리포트**: HTML 리포트와 차트로 결과 시각화

## 🚀 빠른 시작

### 1. 설치

```bash
# 저장소 클론
git clone https://github.com/sacross93/kt-prompt.git
cd kt-prompt

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정

```bash
# 설정 파일 생성
python cli.py create-config

# .env 파일 편집하여 API 키 설정
# GEMINI_API_KEY=your_api_key_here
```

### 3. 샘플 프롬프트 생성

```bash
# 샘플 프롬프트 생성
python cli.py create-prompt --output initial_prompt.txt
```

### 4. 최적화 실행

```bash
# 기본 최적화 실행
python cli.py optimize --initial-prompt initial_prompt.txt

# 고급 옵션으로 실행
python cli.py optimize \
  --initial-prompt initial_prompt.txt \
  --target-accuracy 0.98 \
  --max-iterations 15 \
  --csv-path data/samples.csv
```

## 📋 요구사항

- Python 3.8+
- Google Generative AI API 키
- 필수 패키지: `google-generativeai`, `pandas`, `python-dotenv`

## 🛠️ 설치 가이드

### 1. Python 환경 설정

```bash
# Python 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 2. 패키지 설치

```bash
pip install -r requirements.txt
```

### 3. API 키 설정

1. [Google AI Studio](https://makersuite.google.com/app/apikey)에서 API 키 생성
2. `.env` 파일에 API 키 추가:

```env
GEMINI_API_KEY=your_actual_api_key_here
TARGET_ACCURACY=0.95
MAX_ITERATIONS=10
BATCH_SIZE=50
```

## 📖 사용법

### 명령행 인터페이스

#### 기본 명령어

```bash
# 도움말 보기
python cli.py --help

# API 연결 테스트
python cli.py test-api

# 데이터셋 분석
python cli.py analyze-dataset --csv-path data/samples.csv

# 설정 파일 생성
python cli.py create-config --output .env

# 샘플 프롬프트 생성
python cli.py create-prompt --output prompt.txt
```

#### 최적화 실행

```bash
# 기본 최적화
python cli.py optimize --initial-prompt prompt.txt

# 상세 옵션
python cli.py optimize \
  --initial-prompt prompt.txt \
  --target-accuracy 0.95 \
  --max-iterations 10 \
  --batch-size 50 \
  --csv-path data/samples.csv \
  --output-dir results \
  --log-level INFO
```

### Python API 사용

```python
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

# 설정 생성
config = OptimizationConfig.from_env()
config.target_accuracy = 0.95
config.max_iterations = 10

# 최적화 실행
optimizer = GeminiPromptOptimizer(config)
result = optimizer.run_optimization("initial_prompt.txt")

print(f"최종 정확도: {result.final_accuracy:.2%}")
print(f"총 반복 횟수: {result.total_iterations}")
```

## ⚙️ 설정 옵션

### 환경 변수

| 변수명 | 설명 | 기본값 |
|--------|------|--------|
| `GEMINI_API_KEY` | Gemini API 키 | 필수 |
| `TARGET_ACCURACY` | 목표 정확도 (0.0-1.0) | 0.95 |
| `MAX_ITERATIONS` | 최대 반복 횟수 | 10 |
| `BATCH_SIZE` | 배치 크기 | 50 |
| `API_RETRY_COUNT` | API 재시도 횟수 | 3 |
| `SAMPLES_CSV_PATH` | 샘플 데이터 경로 | data/samples.csv |
| `PROMPT_DIR` | 프롬프트 저장 디렉토리 | prompt |
| `ANALYSIS_DIR` | 분석 결과 저장 디렉토리 | analysis |

### 고급 설정

```python
config = OptimizationConfig(
    gemini_api_key="your_key",
    target_accuracy=0.98,
    max_iterations=15,
    batch_size=25,
    convergence_threshold=0.001,  # 수렴 임계값
    patience=3,  # 조기 종료 인내심
    flash_model="gemini-1.5-flash",
    pro_model="gemini-1.5-pro"
)
```

## 📊 결과 분석

### 출력 파일

최적화 완료 후 다음 파일들이 생성됩니다:

```
analysis/
├── optimization.log              # 실행 로그
├── final_report.txt             # 최종 결과 리포트
├── optimization_metrics_*.json  # 성능 지표
├── analysis_iter_*.txt          # 각 반복별 분석
└── prompt_changes.log           # 프롬프트 변경 이력

prompt/
├── system_prompt_v1.txt         # 개선된 프롬프트들
├── system_prompt_v2.txt
└── system_prompt_final.txt      # 최종 프롬프트

results/
├── optimization_report.html    # HTML 시각화 리포트
├── dashboard.json              # 대시보드 데이터
└── charts/                     # 차트 데이터
```

### HTML 리포트

`optimization_report.html` 파일을 브라우저에서 열면 다음을 확인할 수 있습니다:

- 최적화 진행 상황 차트
- 카테고리별 성능 분석
- 오류 패턴 분석
- 상세 통계 정보

## 🔧 트러블슈팅

### 일반적인 문제

#### API 키 오류
```
❌ API key validation failed
```
**해결방법**: `.env` 파일의 `GEMINI_API_KEY`가 올바른지 확인

#### 할당량 초과
```
❌ API quota exceeded
```
**해결방법**: API 사용량 확인 후 대기 또는 배치 크기 감소

#### 메모리 부족
```
❌ Memory error during processing
```
**해결방법**: `BATCH_SIZE` 값을 줄이거나 시스템 메모리 확인

#### 파일 없음 오류
```
❌ File not found: data/samples.csv
```
**해결방법**: CSV 파일 경로 확인 또는 `--csv-path` 옵션으로 경로 지정

### 성능 최적화

#### 느린 실행 속도
- `BATCH_SIZE` 증가 (API 한도 내에서)
- 캐싱 활성화 확인
- 네트워크 연결 상태 확인

#### 메모리 사용량 최적화
- `BATCH_SIZE` 감소
- 대용량 데이터셋의 경우 청크 단위 처리
- 불필요한 로그 레벨 조정

## 🏗️ 아키텍처

### 시스템 구조

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

### 주요 컴포넌트

- **GeminiFlashClassifier**: 문장 분류 수행
- **GeminiProAnalyzer**: 오류 분석 및 개선 제안
- **PromptOptimizer**: 프롬프트 자동 개선
- **IterationController**: 반복 과정 제어
- **ResultAnalyzer**: 결과 분석 및 통계

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 `LICENSE` 파일을 참조하세요.

## 🆘 지원

문제가 발생하거나 질문이 있으시면:

1. [Issues](../../issues) 페이지에서 기존 이슈 확인
2. 새로운 이슈 생성 시 다음 정보 포함:
   - Python 버전
   - 오류 메시지 전문
   - 실행 환경 (OS, 메모리 등)
   - 재현 가능한 최소 예제

## 🔄 업데이트 로그

### v1.0.0
- 초기 릴리스
- Gemini Flash/Pro 기반 자동 최적화
- 실시간 모니터링 및 시각화
- 성능 최적화 및 캐싱

---

**Made with ❤️ for Korean NLP optimization**