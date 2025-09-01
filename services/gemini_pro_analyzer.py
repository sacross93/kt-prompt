"""
Gemini Pro analyzer for error analysis and prompt improvement suggestions
"""
import os
from datetime import datetime
from typing import List, Dict, Any
import logging
from services.gemini_client import GeminiClient
from models.data_models import ErrorCase, AnalysisReport
from models.exceptions import APIError, FileProcessingError
from utils.file_utils import write_text_file, ensure_directory_exists
from config import OptimizationConfig

logger = logging.getLogger("gemini_optimizer.pro_analyzer")

class GeminiProAnalyzer:
    """Gemini Pro-based error analyzer and prompt improvement advisor"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.client = GeminiClient(config)
        self.model = self.client.get_pro_model()
        
        # Analysis prompt template
        self.analysis_prompt_template = """
당신은 한국어 문장 분류 시스템의 전문 분석가입니다. 
주어진 오류 사례들을 분석하여 시스템 프롬프트의 개선점을 제안해주세요.

분류 기준:
- 유형: 사실형, 추론형, 대화형, 예측형
- 극성: 긍정, 부정, 미정  
- 시제: 과거, 현재, 미래
- 확실성: 확실, 불확실

현재 시스템 프롬프트:
{current_prompt}

오류 사례들:
{error_cases}

다음 형식으로 분석해주세요:

## 오류 패턴 분석
[주요 오류 패턴들을 분석하고 원인을 설명]

## 문제점 식별
[현재 프롬프트의 구체적인 문제점들]

## 개선 제안
[구체적이고 실행 가능한 개선 방안들]

## 프롬프트 수정 방향
[프롬프트에서 수정해야 할 구체적인 부분들]

## 신뢰도 점수
[이 분석의 신뢰도를 0.0-1.0 사이로 평가]
"""
        
        logger.info("GeminiProAnalyzer initialized")
    
    def analyze_errors(self, errors: List[ErrorCase], current_prompt: str) -> AnalysisReport:
        """Analyze errors and generate improvement suggestions"""
        try:
            logger.info(f"Analyzing {len(errors)} error cases")
            
            if not errors:
                return self._create_no_errors_report()
            
            # Prepare error cases text
            error_cases_text = self._format_error_cases(errors)
            
            # Create analysis prompt
            analysis_prompt = self.analysis_prompt_template.format(
                current_prompt=current_prompt,
                error_cases=error_cases_text
            )
            
            # Generate analysis
            response = self.client.generate_content_with_retry(self.model, analysis_prompt)
            
            # Parse analysis response
            analysis_report = self._parse_analysis_response(response, errors)
            
            logger.info(f"Analysis complete with confidence score: {analysis_report.confidence_score:.2f}")
            return analysis_report
            
        except Exception as e:
            logger.error(f"Failed to analyze errors: {e}")
            raise APIError(f"Error analysis failed: {e}")
    
    def save_analysis(self, report: AnalysisReport, file_path: str = None) -> str:
        """Save analysis report to file"""
        try:
            if file_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = os.path.join(self.config.analysis_dir, f"analysis_{timestamp}.txt")
            
            # Ensure directory exists
            ensure_directory_exists(os.path.dirname(file_path))
            
            # Format report content
            report_content = self._format_analysis_report(report)
            
            # Save to file
            write_text_file(file_path, report_content)
            
            logger.info(f"Analysis report saved to: {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Failed to save analysis report: {e}")
            raise FileProcessingError(f"Failed to save analysis: {e}")
    
    def _format_error_cases(self, errors: List[ErrorCase]) -> str:
        """Format error cases for analysis prompt"""
        if not errors:
            return "오류 사례가 없습니다."
        
        error_lines = []
        
        # Group errors by type for better analysis
        error_by_type = {}
        for error in errors:
            error_types = error.error_type.split(',')
            for error_type in error_types:
                if error_type not in error_by_type:
                    error_by_type[error_type] = []
                error_by_type[error_type].append(error)
        
        for error_type, error_list in error_by_type.items():
            error_lines.append(f"\n=== {error_type.upper()} 오류 ({len(error_list)}개) ===")
            
            # Show up to 5 examples of each error type
            for i, error in enumerate(error_list[:5]):
                error_lines.append(f"\n{i+1}. 문장 ID {error.question_id}:")
                error_lines.append(f"   문장: {error.sentence}")
                error_lines.append(f"   예상: {error.expected}")
                error_lines.append(f"   실제: {error.predicted}")
                
                # Show error details
                error_details = error.get_error_details()
                if error_details and "format_error" not in error_details:
                    error_lines.append("   오류 세부사항:")
                    for category, detail in error_details.items():
                        error_lines.append(f"     - {category}: {detail}")
            
            if len(error_list) > 5:
                error_lines.append(f"   ... 및 {len(error_list) - 5}개 추가 사례")
        
        return "\n".join(error_lines)
    
    def _parse_analysis_response(self, response: str, errors: List[ErrorCase]) -> AnalysisReport:
        """Parse analysis response into AnalysisReport"""
        try:
            # Extract sections from response
            sections = self._extract_sections(response)
            
            # Extract error patterns
            error_patterns = self._extract_error_patterns(errors)
            
            # Extract improvement suggestions
            improvement_suggestions = self._extract_improvements(sections.get("개선 제안", ""))
            
            # Extract prompt modifications
            prompt_modifications = self._extract_modifications(sections.get("프롬프트 수정 방향", ""))
            
            # Extract confidence score
            confidence_score = self._extract_confidence_score(sections.get("신뢰도 점수", "0.5"))
            
            return AnalysisReport(
                total_errors=len(errors),
                error_patterns=error_patterns,
                improvement_suggestions=improvement_suggestions,
                prompt_modifications=prompt_modifications,
                confidence_score=confidence_score,
                analysis_text=response
            )
            
        except Exception as e:
            logger.warning(f"Failed to parse analysis response, using fallback: {e}")
            return self._create_fallback_report(response, errors)
    
    def _extract_sections(self, response: str) -> Dict[str, str]:
        """Extract sections from analysis response"""
        sections = {}
        current_section = None
        current_content = []
        
        lines = response.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Check if line is a section header
            if line.startswith('##') or line.startswith('#'):
                # Save previous section
                if current_section:
                    sections[current_section] = '\n'.join(current_content).strip()
                
                # Start new section
                current_section = line.replace('#', '').strip()
                current_content = []
            else:
                if current_section:
                    current_content.append(line)
        
        # Save last section
        if current_section:
            sections[current_section] = '\n'.join(current_content).strip()
        
        return sections
    
    def _extract_error_patterns(self, errors: List[ErrorCase]) -> Dict[str, int]:
        """Extract error patterns from error cases"""
        patterns = {}
        
        for error in errors:
            error_types = error.error_type.split(',')
            for error_type in error_types:
                patterns[error_type] = patterns.get(error_type, 0) + 1
        
        return patterns
    
    def _extract_improvements(self, improvement_text: str) -> List[str]:
        """Extract improvement suggestions from text"""
        improvements = []
        
        lines = improvement_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                # Remove bullet point and clean up
                improvement = line[1:].strip()
                if improvement:
                    improvements.append(improvement)
            elif line and len(line) > 10:  # Assume non-bullet lines are also suggestions
                improvements.append(line)
        
        # If no bullet points found, split by sentences
        if not improvements and improvement_text:
            sentences = improvement_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    improvements.append(sentence)
        
        return improvements[:10]  # Limit to 10 suggestions
    
    def _extract_modifications(self, modification_text: str) -> List[str]:
        """Extract prompt modifications from text"""
        modifications = []
        
        lines = modification_text.split('\n')
        for line in lines:
            line = line.strip()
            if line and (line.startswith('-') or line.startswith('*') or line.startswith('•')):
                modification = line[1:].strip()
                if modification:
                    modifications.append(modification)
            elif line and len(line) > 10:
                modifications.append(line)
        
        if not modifications and modification_text:
            sentences = modification_text.split('.')
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 10:
                    modifications.append(sentence)
        
        return modifications[:10]  # Limit to 10 modifications
    
    def _extract_confidence_score(self, confidence_text: str) -> float:
        """Extract confidence score from text"""
        import re
        
        # Look for decimal numbers between 0 and 1
        matches = re.findall(r'([0-1]?\.\d+)', confidence_text)
        if matches:
            try:
                score = float(matches[0])
                return max(0.0, min(1.0, score))  # Clamp between 0 and 1
            except ValueError:
                pass
        
        # Look for percentages
        matches = re.findall(r'(\d+)%', confidence_text)
        if matches:
            try:
                score = float(matches[0]) / 100.0
                return max(0.0, min(1.0, score))
            except ValueError:
                pass
        
        # Default confidence score
        return 0.7
    
    def _create_no_errors_report(self) -> AnalysisReport:
        """Create report when there are no errors"""
        return AnalysisReport(
            total_errors=0,
            error_patterns={},
            improvement_suggestions=["현재 프롬프트가 모든 테스트 케이스를 정확히 분류했습니다."],
            prompt_modifications=["추가 최적화가 필요하지 않습니다."],
            confidence_score=1.0,
            analysis_text="오류가 없어 분석할 내용이 없습니다."
        )
    
    def _create_fallback_report(self, response: str, errors: List[ErrorCase]) -> AnalysisReport:
        """Create fallback report when parsing fails"""
        error_patterns = self._extract_error_patterns(errors)
        
        return AnalysisReport(
            total_errors=len(errors),
            error_patterns=error_patterns,
            improvement_suggestions=["분석 응답 파싱에 실패했습니다. 수동으로 검토가 필요합니다."],
            prompt_modifications=["원본 분석 결과를 참조하여 수동으로 수정하세요."],
            confidence_score=0.3,
            analysis_text=response
        )
    
    def _format_analysis_report(self, report: AnalysisReport) -> str:
        """Format analysis report for file output"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_lines = [
            "=" * 60,
            "GEMINI PRO 오류 분석 보고서",
            "=" * 60,
            f"생성 시간: {timestamp}",
            f"총 오류 수: {report.total_errors}",
            f"신뢰도 점수: {report.confidence_score:.2f}",
            "",
            "오류 패턴:",
            "-" * 30
        ]
        
        for pattern, count in report.error_patterns.items():
            report_lines.append(f"  {pattern}: {count}개")
        
        report_lines.extend([
            "",
            "개선 제안:",
            "-" * 30
        ])
        
        for i, suggestion in enumerate(report.improvement_suggestions, 1):
            report_lines.append(f"  {i}. {suggestion}")
        
        report_lines.extend([
            "",
            "프롬프트 수정 방향:",
            "-" * 30
        ])
        
        for i, modification in enumerate(report.prompt_modifications, 1):
            report_lines.append(f"  {i}. {modification}")
        
        report_lines.extend([
            "",
            "=" * 60,
            "원본 분석 결과:",
            "=" * 60,
            report.analysis_text
        ])
        
        return "\n".join(report_lines)
    
    def generate_improvement_prompt(self, current_prompt: str, analysis_report: AnalysisReport) -> str:
        """Generate prompt for improving the system prompt"""
        improvement_prompt = f"""
다음은 한국어 문장 분류 시스템의 현재 프롬프트와 오류 분석 결과입니다.
분석 결과를 바탕으로 프롬프트를 개선해주세요.

현재 프롬프트:
{current_prompt}

오류 분석 결과:
- 총 오류 수: {analysis_report.total_errors}
- 주요 오류 패턴: {', '.join(analysis_report.error_patterns.keys())}

개선 제안:
{chr(10).join(f'- {suggestion}' for suggestion in analysis_report.improvement_suggestions)}

수정 방향:
{chr(10).join(f'- {modification}' for modification in analysis_report.prompt_modifications)}

위 분석을 바탕으로 개선된 시스템 프롬프트를 작성해주세요.
기존 프롬프트의 좋은 부분은 유지하면서 문제점만 수정하세요.
"""
        return improvement_prompt