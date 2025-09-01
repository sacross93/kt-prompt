# 트러블슈팅 가이드

Gemini Prompt Optimizer 사용 중 발생할 수 있는 문제들과 해결 방법을 정리한 가이드입니다.

## 목차

- [설치 및 설정 문제](#설치-및-설정-문제)
- [API 관련 문제](#api-관련-문제)
- [데이터 처리 문제](#데이터-처리-문제)
- [성능 및 메모리 문제](#성능-및-메모리-문제)
- [최적화 과정 문제](#최적화-과정-문제)
- [파일 및 권한 문제](#파일-및-권한-문제)

## 설치 및 설정 문제

### Python 버전 호환성 문제

**문제:**
```
ERROR: Package requires Python >=3.8
```

**해결방법:**
1. Python 버전 확인:
   ```bash
   python --version
   ```
2. Python 3.8 이상 설치
3. 가상환경 재생성:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 또는
   venv\Scripts\activate     # Windows
   ```

### 패키지 설치 실패

**문제:**
```
ERROR: Failed building wheel for package
```

**해결방법:**
1. pip 업그레이드:
   ```bash
   pip install --upgrade pip
   ```
2. 개별 패키지 설치:
   ```bash
   pip install google-generativeai
   pip install pandas
   pip install python-dotenv
   ```
3. 시스템 의존성 확인 (Linux):
   ```bash
   sudo apt-get update
   sudo apt-get install python3-dev build-essential
   ```

### 환경 변수 설정 문제

**문제:**
```
❌ Configuration error: GEMINI_API_KEY environment variable is required
```

**해결방법:**
1. `.env` 파일 생성:
   ```bash
   python cli.py create-config
   ```
2. API 키 설정:
   ```env
   GEMINI_API_KEY=your_actual_api_key_here
   ```
3. 파일 위치 확인 (프로젝트 루트 디렉토리)
4. 파일 권한 확인:
   ```bash
   chmod 600 .env
   ```

## API 관련 문제

### API 키 인증 실패

**문제:**
```
❌ API key validation failed
```

**해결방법:**
1. API 키 유효성 확인:
   - [Google AI Studio](https://makersuite.google.com/app/apikey)에서 키 상태 확인
   - 키가 활성화되어 있는지 확인
2. 키 형식 확인:
   ```env
   # 올바른 형식
   GEMINI_API_KEY=AIzaSyD...
   
   # 잘못된 형식 (따옴표 사용 금지)
   GEMINI_API_KEY="AIzaSyD..."
   ```
3. 네트워크 연결 확인:
   ```bash
   python cli.py test-api
   ```

### API 할당량 초과

**문제:**
```
❌ API quota exceeded
```

**해결방법:**
1. 할당량 확인:
   - Google AI Studio에서 사용량 확인
   - 일일/월간 한도 확인
2. 배치 크기 감소:
   ```env
   BATCH_SIZE=25  # 기본값 50에서 감소
   ```
3. 처리 속도 조절:
   ```env
   API_RETRY_COUNT=5  # 재시도 횟수 증가
   ```
4. 대기 후 재실행

### API 응답 시간 초과

**문제:**
```
❌ Network error: timeout
```

**해결방법:**
1. 네트워크 연결 확인
2. 배치 크기 감소:
   ```bash
   python cli.py optimize --batch-size 10
   ```
3. 재시도 설정 조정:
   ```python
   config.api_retry_count = 5
   ```

### 모델 접근 오류

**문제:**
```
❌ Flash model not available
```

**해결방법:**
1. 모델명 확인:
   ```env
   FLASH_MODEL=gemini-1.5-flash
   PRO_MODEL=gemini-1.5-pro
   ```
2. API 키 권한 확인
3. 지역 제한 확인

## 데이터 처리 문제

### CSV 파일 형식 오류

**문제:**
```
❌ Missing required columns: ['type', 'polarity']
```

**해결방법:**
1. CSV 파일 형식 확인:
   ```csv
   id,sentence,type,polarity,tense,certainty
   1,문장,사실형,긍정,현재,확실
   ```
2. 인코딩 확인 (UTF-8 사용)
3. 데이터셋 분석:
   ```bash
   python cli.py analyze-dataset --csv-path your_file.csv
   ```

### 데이터 검증 실패

**문제:**
```
❌ Invalid type 'unknown_type' for sample 1
```

**해결방법:**
1. 유효한 라벨 확인:
   - 유형: 사실형, 추론형, 대화형, 예측형
   - 극성: 긍정, 부정, 미정
   - 시제: 과거, 현재, 미래
   - 확실성: 확실, 불확실
2. 데이터 정리:
   ```python
   # 잘못된 라벨 찾기
   python -c "
   from services.csv_processor import CSVProcessor
   processor = CSVProcessor('data/samples.csv')
   is_valid, errors = processor.validate_dataset()
   print(errors)
   "
   ```

### 빈 문장 오류

**문제:**
```
❌ Empty sentence for sample 5
```

**해결방법:**
1. 빈 문장 제거 또는 수정
2. 데이터 전처리:
   ```python
   import pandas as pd
   df = pd.read_csv('data/samples.csv')
   df = df.dropna(subset=['sentence'])
   df = df[df['sentence'].str.strip() != '']
   df.to_csv('data/samples_clean.csv', index=False)
   ```

## 성능 및 메모리 문제

### 메모리 부족

**문제:**
```
❌ Memory error during processing
```

**해결방법:**
1. 배치 크기 감소:
   ```env
   BATCH_SIZE=10
   ```
2. 메모리 사용량 모니터링:
   ```bash
   pip install psutil
   ```
3. 대용량 데이터 청크 처리:
   ```python
   config.batch_size = 25
   ```

### 처리 속도 저하

**문제:**
- 매우 느린 실행 속도
- 응답 시간 지연

**해결방법:**
1. 캐싱 활성화 확인
2. 배치 크기 최적화:
   ```bash
   # 테스트해보며 최적값 찾기
   python cli.py optimize --batch-size 100  # 큰 배치
   python cli.py optimize --batch-size 25   # 작은 배치
   ```
3. 병렬 처리 비활성화 (API 한도 고려):
   ```python
   # 순차 처리로 변경
   processor = BatchProcessor(max_workers=1)
   ```

### 디스크 공간 부족

**문제:**
```
❌ No space left on device
```

**해결방법:**
1. 디스크 공간 확인:
   ```bash
   df -h
   ```
2. 임시 파일 정리:
   ```bash
   rm -rf cache/
   rm -rf __pycache__/
   ```
3. 로그 파일 정리:
   ```bash
   find analysis/ -name "*.log" -mtime +7 -delete
   ```

## 최적화 과정 문제

### 수렴하지 않는 최적화

**문제:**
- 정확도가 개선되지 않음
- 최대 반복 횟수 도달

**해결방법:**
1. 목표 정확도 조정:
   ```bash
   python cli.py optimize --target-accuracy 0.90
   ```
2. 최대 반복 횟수 증가:
   ```bash
   python cli.py optimize --max-iterations 20
   ```
3. 수렴 설정 조정:
   ```python
   config.convergence_threshold = 0.005
   config.patience = 5
   ```

### 프롬프트 개선 실패

**문제:**
```
❌ Prompt improvement failed
```

**해결방법:**
1. 초기 프롬프트 품질 확인
2. 샘플 프롬프트 사용:
   ```bash
   python cli.py create-prompt --output better_prompt.txt
   ```
3. 수동 프롬프트 수정 후 재시도

### 분석 결과 파싱 오류

**문제:**
```
❌ Failed to parse analysis response
```

**해결방법:**
1. Gemini Pro 모델 상태 확인
2. 분석 프롬프트 단순화
3. 재시도 또는 수동 분석

## 파일 및 권한 문제

### 파일 접근 권한 오류

**문제:**
```
❌ Permission denied: analysis/report.txt
```

**해결방법:**
1. 디렉토리 권한 확인:
   ```bash
   ls -la analysis/
   ```
2. 권한 수정:
   ```bash
   chmod 755 analysis/
   chmod 644 analysis/*
   ```
3. 소유자 확인:
   ```bash
   chown -R $USER:$USER analysis/
   ```

### 파일 경로 문제

**문제:**
```
❌ File not found: data/samples.csv
```

**해결방법:**
1. 현재 디렉토리 확인:
   ```bash
   pwd
   ls -la
   ```
2. 절대 경로 사용:
   ```bash
   python cli.py optimize --csv-path /full/path/to/samples.csv
   ```
3. 파일 존재 확인:
   ```bash
   find . -name "samples.csv"
   ```

### 디렉토리 생성 실패

**문제:**
```
❌ Failed to create directory
```

**해결방법:**
1. 상위 디렉토리 권한 확인
2. 수동 디렉토리 생성:
   ```bash
   mkdir -p prompt analysis cache
   ```
3. 쓰기 권한 확인:
   ```bash
   touch test_file && rm test_file
   ```

## 로그 및 디버깅

### 상세 로그 활성화

문제 진단을 위해 상세 로그를 활성화하세요:

```bash
python cli.py optimize --log-level DEBUG --initial-prompt prompt.txt
```

### 로그 파일 위치

로그 파일들의 위치:
- `analysis/optimization.log`: 메인 로그
- `analysis/prompt_changes.log`: 프롬프트 변경 이력
- `analysis/analysis_iter_*.txt`: 각 반복별 분석

### 디버그 정보 수집

문제 보고 시 다음 정보를 포함하세요:

```bash
# 시스템 정보
python --version
pip list | grep -E "(google|pandas|dotenv)"

# 설정 정보 (API 키 제외)
python -c "
from config import OptimizationConfig
try:
    config = OptimizationConfig.from_env()
    print(f'Target: {config.target_accuracy}')
    print(f'Max iter: {config.max_iterations}')
    print(f'Batch size: {config.batch_size}')
except Exception as e:
    print(f'Config error: {e}')
"

# API 연결 테스트
python cli.py test-api
```

## 자주 묻는 질문 (FAQ)

### Q: 최적화가 너무 오래 걸려요

**A:** 다음을 시도해보세요:
- 배치 크기 증가 (`--batch-size 100`)
- 목표 정확도 낮추기 (`--target-accuracy 0.90`)
- 최대 반복 횟수 제한 (`--max-iterations 5`)

### Q: 정확도가 개선되지 않아요

**A:** 다음을 확인해보세요:
- 초기 프롬프트 품질
- 데이터셋 품질 및 일관성
- 목표 정확도가 현실적인지
- 수렴 설정 조정

### Q: API 비용이 걱정돼요

**A:** 비용 절약 방법:
- 작은 배치 크기 사용
- 캐싱 활성화 (기본 활성화됨)
- 테스트용 작은 데이터셋 사용
- 목표 정확도를 적절히 설정

### Q: 결과를 어떻게 해석하나요?

**A:** 다음 파일들을 확인하세요:
- `analysis/final_report.txt`: 최종 결과 요약
- `optimization_report.html`: 시각화된 리포트
- `analysis/optimization.log`: 상세 실행 로그

## 추가 도움

문제가 해결되지 않으면:

1. [GitHub Issues](../../issues)에서 유사한 문제 검색
2. 새 이슈 생성 시 다음 정보 포함:
   - 운영체제 및 Python 버전
   - 전체 오류 메시지
   - 실행한 명령어
   - 설정 파일 내용 (API 키 제외)
   - 로그 파일 내용

3. 디버그 정보 수집:
   ```bash
   python cli.py optimize --log-level DEBUG --initial-prompt prompt.txt > debug.log 2>&1
   ```