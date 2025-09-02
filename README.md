# 🤖 KT Prompt Optimizer

Gemini 2.5 Flash와 Pro를 활용한 자동 프롬프트 최적화 시스템입니다. 한국어 문장 분류 문제를 해결하기 위해 프롬프트를 자동으로 분석하고 개선합니다.

## ✨ 주요 기능

### 🤖 자동화된 최적화 시스템 (NEW!)
- **3단계 자동 최적화**: 정확도 → 한글비율 → 길이압축
- **Gemini Pro 기반 분석**: 지능적 오답 분석 및 프롬프트 개선
- **KT 점수 최적화**: 0.8×정확도 + 0.1×한글비율 + 0.1×길이점수
- **실시간 모니터링**: 각 단계별 성능 변화 추적

### 🔧 기존 기능
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

#### KT 해커톤 특화 최적화 (권장)
```bash
# 자동화된 3단계 최적화 (Gemini Pro 기반)
python kt_hackathon_cli.py optimize --prompt your_prompt.txt --auto --target 0.9

# 기본 3단계 최적화
python kt_hackathon_cli.py optimize --prompt your_prompt.txt --target 0.9

# KT 점수 계산
python kt_hackathon_cli.py score --prompt your_prompt.txt --accuracy 0.75
```

#### 기존 최적화 시스템
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

## 🤖 자동화된 최적화 시스템 완전 가이드

### 🎯 KT 해커톤 특화 기능
- **3단계 자동 최적화**: 정확도 → 한글비율 → 길이압축
- **KT 점수 공식**: 0.8×정확도 + 0.1×한글비율 + 0.1×길이점수
- **Gemini Pro 기반**: 지능적 오답 분석 및 프롬프트 개선
- **실시간 모니터링**: 각 단계별 성능 변화 추적
- **완전 자동화**: 목표 달성까지 사용자 개입 없이 자동 실행

### 🚀 빠른 시작 (자동화 시스템)

#### 1단계: 기본 설정
```bash
# API 키 설정 (.env 파일)
GEMINI_API_KEY=your_api_key_here

# 프로젝트 디렉토리로 이동
cd kt-prompt
```

#### 2단계: 자동화된 최적화 실행
```bash
# 🤖 완전 자동화 모드 (권장)
python kt_hackathon_cli.py optimize --prompt your_prompt.txt --auto --target 0.9

# 📊 기본 3단계 최적화
python kt_hackathon_cli.py optimize --prompt your_prompt.txt --target 0.9

# 🧮 KT 점수만 계산
python kt_hackathon_cli.py score --prompt your_prompt.txt --accuracy 0.75
```

### 📋 명령어 옵션 상세 가이드

#### optimize 명령어
```bash
python kt_hackathon_cli.py optimize [옵션들]

필수 옵션:
  --prompt PATH          초기 프롬프트 파일 경로

선택 옵션:
  --auto                 자동화된 최적화 사용 (Gemini Pro 기반)
  --target FLOAT         목표 KT 점수 (기본값: 0.9)
  --samples PATH         샘플 데이터 경로 (기본값: data/samples.csv)
  --output PATH          출력 디렉토리 (기본값: prompt/gemini)
```

#### score 명령어
```bash
python kt_hackathon_cli.py score --prompt PATH --accuracy FLOAT

예시:
python kt_hackathon_cli.py score --prompt my_prompt.txt --accuracy 0.85
```

#### 기타 유용한 명령어
```bash
# 📊 샘플 데이터 분석
python kt_hackathon_cli.py analyze-data --samples data/samples.csv

# 🧪 프롬프트 정확도 테스트
python kt_hackathon_cli.py test --prompt your_prompt.txt --samples data/samples.csv
```

### 🔄 자동화 프로세스 상세 설명

#### 1단계: 자동 정확도 최적화 (목표: 0.8+)
```
🔄 반복 프로세스:
1. Gemini 2.5 Flash로 현재 정확도 측정
2. Gemini 2.5 Pro로 오답 패턴 분석
3. Gemini 2.5 Pro로 개선된 프롬프트 생성
4. 목표 달성까지 자동 반복 (최대 10회)

📊 실시간 출력 예시:
=== 반복 1/10 시작 ===
현재 정확도: 0.4800 (목표: 0.8000)
오답 분석 완료 - 패턴: 3개, 제안: 5개
개선된 프롬프트 생성: auto_optimized_v1.txt
```

#### 2단계: 한글 비율 최적화 (목표: 90%+)
```
🔧 최적화 과정:
- 영어 표현을 자연스러운 한글로 번역
- 전문용어의 적절한 한글화
- 문체 통일 및 가독성 향상

📊 출력 예시:
현재 한글 비율: 0.6615, 목표: 0.9000
최적화 후 한글 비율: 0.9123
```

#### 3단계: 자동 길이 압축 (목표: 3000자 이하)
```
🗜️ 압축 전략:
- Gemini 2.5 Pro의 지능적 압축
- 핵심 내용 보존 (정확도 손실 5% 이내)
- 중복 제거 및 간결한 표현

📊 출력 예시:
원본 길이: 3542자 → 압축 후: 2987자 (84.3%)
정확도 변화: +0.0120 (개선됨!)
```

### 📊 실행 결과 예시

#### 성공적인 최적화 완료
```bash
🏆 KT 해커톤 프롬프트 최적화 시스템
============================================================
📊 KT 점수 공식: 0.8×정확도 + 0.1×한글비율 + 0.1×길이점수
🎯 목표: 총점 0.9점 이상 달성
============================================================

🤖 자동화된 3단계 최적화 시작... (Gemini Pro 기반)

[1단계] 자동 정확도 최적화 (5회 반복)
  반복 1: 0.4800 → 0.5200 (+0.0400)
  반복 2: 0.5200 → 0.6100 (+0.0900)
  반복 3: 0.6100 → 0.7300 (+0.1200)
  반복 4: 0.7300 → 0.8100 (+0.0800)
  ✅ 목표 달성! (0.8100 ≥ 0.8000)

[2단계] 한글 비율 최적화
  현재: 66.15% → 최적화 후: 91.23% ✅

[3단계] 자동 길이 압축
  3542자 → 2987자 (15.7% 압축) ✅

🎉 최종 결과:
📊 최종 KT 점수: 0.9234 / 1.0000
✅ 목표 달성! (0.9234 ≥ 0.9000)

💾 저장된 결과물:
  - 최종 프롬프트: final_optimized_20250902_170530.txt
  - 상세 리포트: optimization_report_20250902_170530.md
  - 최적화 히스토리: optimization_history_20250902_170530.json
  - 요약 정보: summary_20250902_170530.json
```

### 📁 생성되는 파일들

#### 자동 최적화 결과물
```
prompt/auto_optimized/
├── auto_optimized_v1.txt                    # 1차 개선 프롬프트
├── auto_optimized_v2.txt                    # 2차 개선 프롬프트
├── auto_optimized_v3.txt                    # 3차 개선 프롬프트
├── final_optimized_20250902_170530.txt      # 🎯 최종 프롬프트
├── optimization_report_20250902_170530.md   # 📊 상세 리포트
├── optimization_history_20250902_170530.json # 📈 최적화 히스토리
└── summary_20250902_170530.json            # 📋 요약 정보
```

#### 3단계 최적화 결과물
```
prompt/gemini/
├── kt_phase1_accuracy.txt                   # 1단계: 정확도 최적화
├── kt_phase2_korean.txt                     # 2단계: 한글 비율 최적화
├── kt_phase3_final.txt                      # 3단계: 길이 압축 (최종)
└── kt_optimization_report.md                # 3단계 종합 리포트
```

### 🔧 고급 사용법

#### 세밀한 제어
```bash
# 목표 점수 조정
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --auto --target 0.95

# 다른 샘플 데이터 사용
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --auto --samples my_data.csv

# 출력 디렉토리 지정
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --auto --output results/
```

#### 배치 처리
```bash
# 여러 프롬프트 순차 최적화
for prompt in prompt1.txt prompt2.txt prompt3.txt; do
    python kt_hackathon_cli.py optimize --prompt $prompt --auto --target 0.9
done
```

### ⚠️ 주의사항 및 팁

#### API 할당량 관리
```
🚨 Gemini Pro 무료 티어 제한:
- 분당 2회 요청
- 자동 최적화 시 시간이 오래 걸릴 수 있음
- 할당량 초과 시 자동으로 대기

💡 팁:
- 유료 플랜 사용 시 더 빠른 최적화 가능
- 기본 3단계 최적화는 할당량 사용량이 적음
```

#### 성능 최적화 팁
```bash
# 1. 작은 샘플로 빠른 테스트
python kt_hackathon_cli.py test --prompt my_prompt.txt --samples small_data.csv

# 2. KT 점수만 빠르게 확인
python kt_hackathon_cli.py score --prompt my_prompt.txt --accuracy 0.8

# 3. 기본 최적화로 빠른 개선
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --target 0.85
```

### 🎯 실전 활용 시나리오

#### 시나리오 1: 처음 사용하는 경우
```bash
# 1. 현재 프롬프트 성능 확인
python kt_hackathon_cli.py test --prompt my_prompt.txt

# 2. 기본 최적화로 빠른 개선
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --target 0.8

# 3. 만족스럽지 않으면 자동화 시스템 사용
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --auto --target 0.9
```

#### 시나리오 2: 높은 점수가 필요한 경우
```bash
# 자동화 시스템으로 최고 성능 추구
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --auto --target 0.95
```

#### 시나리오 3: 빠른 결과가 필요한 경우
```bash
# 기본 3단계 최적화 (빠름)
python kt_hackathon_cli.py optimize --prompt my_prompt.txt --target 0.85
```

### 📚 추가 자료
- 📖 [자동화 최적화 상세 가이드](docs/AUTO_OPTIMIZATION_GUIDE.md)
- 🔧 [문제 해결 가이드](docs/TROUBLESHOOTING.md)
- 📊 [API 참조 문서](docs/API_REFERENCE.md)
- 💡 [사용 예시 모음](docs/EXAMPLES.md)

## 🔄 업데이트 로그

### v2.0.0 (NEW!)
- 🤖 **자동화된 3단계 최적화 시스템** 추가
- 📊 **KT 점수 계산기** 및 모니터링 시스템
- 🎯 **KT 해커톤 특화** CLI 도구
- 📈 **실시간 성능 추적** 및 시각화
- 📝 **종합 최적화 리포트** 자동 생성

### v1.0.0
- 초기 릴리스
- Gemini Flash/Pro 기반 자동 최적화
- 실시간 모니터링 및 시각화
- 성능 최적화 및 캐싱

---

**Made with ❤️ for Korean NLP optimization**