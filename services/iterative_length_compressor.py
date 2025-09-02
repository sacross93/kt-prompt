"""
반복적 길이 압축 최적화기

Gemini 2.5 Pro를 사용해서 내용 손실 없이 덴스하게 압축
"""

import asyncio
import os
from typing import Tuple
import logging

from .gemini_client import GeminiClient
from config import OptimizationConfig

logger = logging.getLogger(__name__)

class IterativeLengthCompressor:
    """반복적 길이 압축 최적화기"""
    
    def __init__(self):
        self.config = OptimizationConfig.from_env()
        self.gemini_client = GeminiClient(self.config)
        self.pro_model = self.gemini_client.get_pro_model()
        
    async def compress_with_pro(self, prompt: str, target_length: int = 3000) -> str:
        """Gemini 2.5 Pro를 사용한 지능적 압축"""
        current_length = len(prompt)
        
        if current_length <= target_length:
            logger.info(f"이미 목표 길이 이하: {current_length}자 <= {target_length}자")
            return prompt
        
        logger.info(f"🗜️ 압축 시작: {current_length}자 → {target_length}자 목표")
        
        # Gemini 2.5 Pro 압축 프롬프트
        compression_prompt = f"""
당신은 프롬프트 압축 전문가입니다.

다음 한국어 문장 분류 프롬프트를 {target_length}자 이하로 압축해주세요:

현재 프롬프트 ({current_length}자):
```
{prompt}
```

압축 요구사항:
1. 핵심 분류 기준과 규칙은 절대 손실하지 말 것
2. 예시는 가장 중요한 것만 2-3개 유지
3. 중복되는 설명 제거
4. 문장을 간결하게 재작성
5. 출력 형식 지침은 반드시 유지
6. 분류 정확도에 영향을 주지 않도록 주의

목표: {target_length}자 이하로 압축하되 분류 성능은 유지

압답 형식:
## 압축 전략
[어떤 부분을 어떻게 압축했는지 설명]

## 압축된 프롬프트
```
[압축된 프롬프트 전체 내용]
```
"""
        
        try:
            # Gemini 2.5 Pro로 압축 요청
            response = self.gemini_client.generate_content_with_retry(
                self.pro_model, compression_prompt
            )
            
            # 압축된 프롬프트 추출
            compressed_prompt = self._extract_compressed_prompt(response)
            
            compressed_length = len(compressed_prompt)
            compression_ratio = compressed_length / current_length
            
            logger.info(f"✅ 압축 완료: {compressed_length}자 (압축률: {compression_ratio:.1%})")
            
            # 압축 결과 저장
            await self._save_compression_result(prompt, compressed_prompt, response)
            
            return compressed_prompt
            
        except Exception as e:
            logger.error(f"Pro 압축 실패: {e}")
            return prompt  # 실패 시 원본 반환
    
    def _extract_compressed_prompt(self, response: str) -> str:
        """응답에서 압축된 프롬프트 추출"""
        try:
            # ``` 블록에서 프롬프트 추출
            lines = response.split('\n')
            in_prompt = False
            prompt_lines = []
            
            for line in lines:
                if line.strip().startswith("```"):
                    if in_prompt:
                        break  # 프롬프트 블록 끝
                    else:
                        in_prompt = True  # 프롬프트 블록 시작
                        continue
                
                if in_prompt:
                    prompt_lines.append(line)
            
            compressed_prompt = '\n'.join(prompt_lines).strip()
            
            # 프롬프트가 비어있으면 전체 응답에서 추출 시도
            if not compressed_prompt:
                # "압축된 프롬프트" 이후 내용 추출
                prompt_start = response.find("압축된 프롬프트")
                if prompt_start != -1:
                    compressed_prompt = response[prompt_start:].strip()
                else:
                    compressed_prompt = response  # 전체 응답 사용
            
            return compressed_prompt
            
        except Exception as e:
            logger.error(f"프롬프트 추출 실패: {e}")
            return response  # 실패 시 전체 응답 반환
    
    async def _save_compression_result(self, original: str, compressed: str, analysis: str):
        """압축 결과 저장"""
        try:
            os.makedirs("prompt/gemini/compression", exist_ok=True)
            
            # 압축 결과 저장
            timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 원본
            with open(f"prompt/gemini/compression/original_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(original)
            
            # 압축본
            with open(f"prompt/gemini/compression/compressed_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(compressed)
            
            # 분석 내용
            with open(f"prompt/gemini/compression/analysis_{timestamp}.txt", 'w', encoding='utf-8') as f:
                f.write(analysis)
            
            logger.info(f"💾 압축 결과 저장 완료: compression_{timestamp}")
            
        except Exception as e:
            logger.error(f"압축 결과 저장 실패: {e}")