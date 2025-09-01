"""
Prompt optimization service for improving system prompts
"""
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import logging
from services.gemini_client import GeminiClient
from services.gemini_pro_analyzer import GeminiProAnalyzer
from models.data_models import AnalysisReport
from models.exceptions import FileProcessingError, PromptOptimizationError
from utils.file_utils import (
    read_text_file, write_text_file, backup_file, 
    get_next_version_filename, ensure_directory_exists
)
from config import OptimizationConfig

logger = logging.getLogger("gemini_optimizer.prompt_optimizer")

class PromptOptimizer:
    """Prompt optimization service using Gemini Pro"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.client = GeminiClient(config)
        self.analyzer = GeminiProAnalyzer(config)
        self.model = self.client.get_pro_model()
        
        # Prompt improvement template
        self.improvement_template = """
당신은 한국어 문장 분류 시스템의 프롬프트 최적화 전문가입니다.
주어진 분석 결과를 바탕으로 시스템 프롬프트를 개선해주세요.

분류 기준:
- 유형: 사실형(객관적 사실), 추론형(분석/의견), 대화형(구어체/인용문), 예측형(미래 예측)
- 극성: 긍정(긍정적/중립적), 부정(부정적/문제점), 미정(질문/불확실)
- 시제: 과거(과거 시제), 현재(현재 시제/일반적 사실), 미래(미래 시제/계획)
- 확실성: 확실(명확한 사실), 불확실(추측/가능성)

현재 프롬프트:
{current_prompt}

분석 결과:
{analysis_summary}

개선 요구사항:
{improvement_requirements}

다음 지침을 따라 프롬프트를 개선해주세요:
1. 기존 프롬프트의 좋은 부분은 유지
2. 분석에서 지적된 문제점만 구체적으로 수정
3. 한글 비율을 최대한 높게 유지
4. 명확하고 구체적인 분류 기준 제시
5. 애매한 경계 사례에 대한 명확한 지침 추가

개선된 시스템 프롬프트만 출력해주세요:
"""
        
        logger.info("PromptOptimizer initialized")
    
    def load_current_prompt(self, file_path: str) -> str:
        """Load current prompt from file"""
        try:
            if not os.path.exists(file_path):
                raise FileProcessingError(f"Prompt file not found: {file_path}")
            
            prompt = read_text_file(file_path)
            logger.info(f"Loaded prompt from {file_path} ({len(prompt)} characters)")
            return prompt
            
        except Exception as e:
            logger.error(f"Failed to load prompt: {e}")
            raise FileProcessingError(f"Failed to load prompt: {e}")
    
    def apply_improvements(self, current_prompt: str, analysis_report: AnalysisReport) -> str:
        """Apply improvements to prompt based on analysis report"""
        try:
            logger.info("Applying improvements to prompt based on analysis")
            
            # Prepare improvement prompt
            analysis_summary = self._create_analysis_summary(analysis_report)
            improvement_requirements = self._create_improvement_requirements(analysis_report)
            
            improvement_prompt = self.improvement_template.format(
                current_prompt=current_prompt,
                analysis_summary=analysis_summary,
                improvement_requirements=improvement_requirements
            )
            
            # Generate improved prompt
            improved_prompt = self.client.generate_content_with_retry(
                self.model, improvement_prompt
            )
            
            # Clean and validate improved prompt
            improved_prompt = self._clean_prompt(improved_prompt)
            
            # Validate improved prompt
            if not self._validate_prompt(improved_prompt):
                logger.warning("Generated prompt failed validation, applying manual fixes")
                improved_prompt = self._apply_manual_fixes(current_prompt, analysis_report)
            
            logger.info(f"Generated improved prompt ({len(improved_prompt)} characters)")
            return improved_prompt
            
        except Exception as e:
            logger.error(f"Failed to apply improvements: {e}")
            raise PromptOptimizationError(f"Prompt improvement failed: {e}")
    
    def save_new_version(self, prompt: str, base_file_path: str = None, version: int = None) -> str:
        """Save new version of prompt"""
        try:
            if base_file_path is None:
                base_file_path = os.path.join(self.config.prompt_dir, "system_prompt.txt")
            
            # Ensure directory exists
            ensure_directory_exists(os.path.dirname(base_file_path))
            
            if version is None:
                # Generate next version filename
                new_file_path = get_next_version_filename(base_file_path)
            else:
                # Use specific version
                directory = os.path.dirname(base_file_path)
                base_name = os.path.splitext(os.path.basename(base_file_path))[0]
                extension = os.path.splitext(base_file_path)[1]
                new_file_path = os.path.join(directory, f"{base_name}_v{version}{extension}")
            
            # Save prompt
            write_text_file(new_file_path, prompt)
            
            logger.info(f"Saved new prompt version: {new_file_path}")
            return new_file_path
            
        except Exception as e:
            logger.error(f"Failed to save new prompt version: {e}")
            raise FileProcessingError(f"Failed to save prompt: {e}")
    
    def log_changes(self, changes: List[str], version: int, log_file: str = None) -> None:
        """Log changes made to prompt"""
        try:
            if log_file is None:
                log_file = os.path.join(self.config.analysis_dir, "prompt_changes.log")
            
            ensure_directory_exists(os.path.dirname(log_file))
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            log_entry = [
                f"\n{'='*50}",
                f"Prompt Version {version} Changes",
                f"Timestamp: {timestamp}",
                f"{'='*50}",
                ""
            ]
            
            for i, change in enumerate(changes, 1):
                log_entry.append(f"{i}. {change}")
            
            log_entry.append("")
            
            # Append to log file
            with open(log_file, 'a', encoding='utf-8') as f:
                f.write('\n'.join(log_entry))
            
            logger.info(f"Logged {len(changes)} changes to {log_file}")
            
        except Exception as e:
            logger.error(f"Failed to log changes: {e}")
    
    def backup_current_prompt(self, file_path: str) -> str:
        """Create backup of current prompt"""
        try:
            backup_path = backup_file(file_path)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to backup prompt: {e}")
            raise FileProcessingError(f"Failed to backup prompt: {e}")
    
    def _create_analysis_summary(self, analysis_report: AnalysisReport) -> str:
        """Create summary of analysis report"""
        summary_lines = [
            f"총 오류 수: {analysis_report.total_errors}",
            f"신뢰도 점수: {analysis_report.confidence_score:.2f}",
            ""
        ]
        
        if analysis_report.error_patterns:
            summary_lines.append("주요 오류 패턴:")
            for pattern, count in analysis_report.error_patterns.items():
                summary_lines.append(f"  - {pattern}: {count}개")
            summary_lines.append("")
        
        return "\n".join(summary_lines)
    
    def _create_improvement_requirements(self, analysis_report: AnalysisReport) -> str:
        """Create improvement requirements from analysis"""
        requirements = []
        
        # Add improvement suggestions
        if analysis_report.improvement_suggestions:
            requirements.append("개선 제안:")
            for suggestion in analysis_report.improvement_suggestions:
                requirements.append(f"  - {suggestion}")
            requirements.append("")
        
        # Add prompt modifications
        if analysis_report.prompt_modifications:
            requirements.append("프롬프트 수정 방향:")
            for modification in analysis_report.prompt_modifications:
                requirements.append(f"  - {modification}")
            requirements.append("")
        
        return "\n".join(requirements)
    
    def _clean_prompt(self, prompt: str) -> str:
        """Clean and normalize prompt text"""
        # Remove extra whitespace
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)  # Multiple newlines to double
        prompt = re.sub(r'[ \t]+', ' ', prompt)  # Multiple spaces to single
        
        # Remove common prefixes that might be added by the model
        prefixes_to_remove = [
            "개선된 시스템 프롬프트:",
            "시스템 프롬프트:",
            "다음은 개선된 프롬프트입니다:",
            "개선된 프롬프트:"
        ]
        
        for prefix in prefixes_to_remove:
            if prompt.strip().startswith(prefix):
                prompt = prompt.strip()[len(prefix):].strip()
        
        # Remove markdown formatting if present
        prompt = re.sub(r'^```.*?\n', '', prompt, flags=re.MULTILINE)
        prompt = re.sub(r'\n```$', '', prompt)
        
        return prompt.strip()
    
    def _validate_prompt(self, prompt: str) -> bool:
        """Validate prompt quality and content"""
        if not prompt or len(prompt) < 100:
            logger.warning("Prompt too short")
            return False
        
        # Check for essential components
        essential_keywords = [
            "분류", "유형", "극성", "시제", "확실성",
            "사실형", "추론형", "대화형", "예측형"
        ]
        
        missing_keywords = []
        for keyword in essential_keywords:
            if keyword not in prompt:
                missing_keywords.append(keyword)
        
        if missing_keywords:
            logger.warning(f"Prompt missing essential keywords: {missing_keywords}")
            return False
        
        # Check Korean character ratio
        korean_chars = sum(1 for char in prompt if ord(char) >= 0xAC00 and ord(char) <= 0xD7A3)
        total_chars = len(prompt)
        korean_ratio = korean_chars / total_chars if total_chars > 0 else 0
        
        if korean_ratio < 0.7:  # At least 70% Korean
            logger.warning(f"Korean character ratio too low: {korean_ratio:.2f}")
            return False
        
        return True
    
    def _apply_manual_fixes(self, current_prompt: str, analysis_report: AnalysisReport) -> str:
        """Apply manual fixes when automatic improvement fails"""
        logger.info("Applying manual fixes to prompt")
        
        improved_prompt = current_prompt
        
        # Apply specific fixes based on error patterns
        error_patterns = analysis_report.error_patterns
        
        # Fix type classification issues
        if any("type" in pattern for pattern in error_patterns.keys()):
            improved_prompt = self._fix_type_classification(improved_prompt)
        
        # Fix polarity classification issues
        if any("polarity" in pattern for pattern in error_patterns.keys()):
            improved_prompt = self._fix_polarity_classification(improved_prompt)
        
        # Fix tense classification issues
        if any("tense" in pattern for pattern in error_patterns.keys()):
            improved_prompt = self._fix_tense_classification(improved_prompt)
        
        # Fix certainty classification issues
        if any("certainty" in pattern for pattern in error_patterns.keys()):
            improved_prompt = self._fix_certainty_classification(improved_prompt)
        
        return improved_prompt
    
    def _fix_type_classification(self, prompt: str) -> str:
        """Fix type classification issues in prompt"""
        # Add more specific type classification guidelines
        type_fix = """
유형 분류 세부 기준:
- 사실형: 객관적 사실, 통계, 역사적 사건, 정의, 뉴스 보도
- 추론형: 분석, 해석, 의견, 추측, "~것 같다", "~할 수 있다"
- 대화형: 직접 인용문, "~습니다", "~해요", 구어체 표현
- 예측형: 미래 예측, 계획, "~할 것이다", "~예정", 날씨 예보
"""
        
        # Insert after the main classification criteria
        if "유형:" in prompt:
            prompt = prompt.replace("유형:", f"유형:\n{type_fix}")
        
        return prompt
    
    def _fix_polarity_classification(self, prompt: str) -> str:
        """Fix polarity classification issues in prompt"""
        polarity_fix = """
극성 분류 세부 기준:
- 긍정: 긍정적 내용, 중립적 서술, 일반적 사실
- 부정: 부정문("~지 않다"), 문제점 지적, 실패, 거부
- 미정: 질문문("~인가?"), 불확실한 추측, 가정법
"""
        
        if "극성:" in prompt:
            prompt = prompt.replace("극성:", f"극성:\n{polarity_fix}")
        
        return prompt
    
    def _fix_tense_classification(self, prompt: str) -> str:
        """Fix tense classification issues in prompt"""
        tense_fix = """
시제 분류 세부 기준:
- 과거: "~했다", "~였다", 과거 시점의 사건
- 현재: "~이다", "~한다", 일반적 사실, 현재 상태
- 미래: "~할 것이다", "~예정", 계획, 미래 예측
"""
        
        if "시제:" in prompt:
            prompt = prompt.replace("시제:", f"시제:\n{tense_fix}")
        
        return prompt
    
    def _fix_certainty_classification(self, prompt: str) -> str:
        """Fix certainty classification issues in prompt"""
        certainty_fix = """
확실성 분류 세부 기준:
- 확실: 명확한 사실, 확정된 내용, 객관적 정보
- 불확실: "~것 같다", "~할 수도", "아마", 추측 표현
"""
        
        if "확실성:" in prompt:
            prompt = prompt.replace("확실성:", f"확실성:\n{certainty_fix}")
        
        return prompt
    
    def calculate_korean_ratio(self, text: str) -> float:
        """Calculate Korean character ratio in text"""
        if not text:
            return 0.0
        
        korean_chars = sum(1 for char in text if ord(char) >= 0xAC00 and ord(char) <= 0xD7A3)
        return korean_chars / len(text)
    
    def optimize_korean_ratio(self, prompt: str) -> str:
        """Optimize Korean character ratio in prompt"""
        # Replace common English terms with Korean equivalents
        replacements = {
            "system": "시스템",
            "prompt": "프롬프트", 
            "classification": "분류",
            "type": "유형",
            "polarity": "극성",
            "tense": "시제",
            "certainty": "확실성",
            "example": "예시",
            "format": "형식"
        }
        
        optimized_prompt = prompt
        for eng, kor in replacements.items():
            optimized_prompt = re.sub(rf'\b{eng}\b', kor, optimized_prompt, flags=re.IGNORECASE)
        
        return optimized_prompt
    
    def get_prompt_statistics(self, prompt: str) -> Dict[str, Any]:
        """Get statistics about prompt"""
        return {
            "total_length": len(prompt),
            "korean_ratio": self.calculate_korean_ratio(prompt),
            "line_count": len(prompt.split('\n')),
            "word_count": len(prompt.split()),
            "has_essential_keywords": self._validate_prompt(prompt)
        }