"""
Advanced parsing handler with multiple strategies and recovery mechanisms
"""
import re
import json
import logging
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from models.exceptions import InvalidResponseError, GeminiOptimizerError

logger = logging.getLogger("gemini_optimizer.advanced_parsing")

@dataclass
class PartialResult:
    """Represents a partial parsing result"""
    parsed_items: List[str]
    failed_items: List[str]
    confidence: float
    parsing_method: str
    errors: List[str]

@dataclass
class ParsedClassification:
    """Represents a successfully parsed classification"""
    index: int
    sentence_type: str
    polarity: str
    tense: str
    certainty: str
    confidence: float = 1.0
    raw_text: str = ""

class AdvancedParsingHandler:
    """Advanced parsing handler with multiple strategies and error recovery"""
    
    def __init__(self):
        # Valid classification values
        self.valid_types = ["사실형", "추론형", "대화형", "예측형"]
        self.valid_polarities = ["긍정", "부정", "미정"]
        self.valid_tenses = ["과거", "현재", "미래"]
        self.valid_certainties = ["확실", "불확실"]
        
        # Alternative representations for fuzzy matching
        self.type_alternatives = {
            "사실": "사실형", "추론": "추론형", "대화": "대화형", "예측": "예측형",
            "fact": "사실형", "inference": "추론형", "dialogue": "대화형", "prediction": "예측형"
        }
        
        self.polarity_alternatives = {
            "positive": "긍정", "negative": "부정", "neutral": "미정",
            "pos": "긍정", "neg": "부정", "neu": "미정"
        }
        
        self.tense_alternatives = {
            "past": "과거", "present": "현재", "future": "미래"
        }
        
        self.certainty_alternatives = {
            "certain": "확실", "uncertain": "불확실", "sure": "확실", "unsure": "불확실"
        }
        
        logger.info("AdvancedParsingHandler initialized")
    
    def try_multiple_parsing_strategies(self, response: str) -> Optional[List[str]]:
        """Try multiple parsing strategies in order of preference"""
        strategies = [
            ("standard_format", self._parse_standard_format),
            ("numbered_list", self._parse_numbered_list),
            ("json_format", self._parse_json_format),
            ("comma_separated", self._parse_comma_separated),
            ("line_by_line", self._parse_line_by_line),
            ("regex_extraction", self._parse_regex_extraction),
            ("fuzzy_matching", self._parse_fuzzy_matching),
            ("intelligent_recovery", self._parse_intelligent_recovery),
            ("context_aware", self._parse_context_aware)
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.debug(f"Trying parsing strategy: {strategy_name}")
                result = strategy_func(response)
                if result and len(result) > 0:
                    logger.info(f"Successfully parsed using strategy: {strategy_name}")
                    return result
            except Exception as e:
                logger.debug(f"Strategy {strategy_name} failed: {e}")
                continue
        
        logger.warning("All parsing strategies failed")
        return None
    
    def normalize_response_format(self, response: str) -> str:
        """Normalize response format for better parsing"""
        if not response:
            return ""
        
        # Remove extra whitespace and normalize line endings
        normalized = re.sub(r'\s+', ' ', response.strip())
        normalized = re.sub(r'\r\n|\r', '\n', normalized)
        
        # Fix common formatting issues
        normalized = re.sub(r'(\d+)\.?\s*([^0-9\n]+)', r'\1. \2', normalized)  # Fix numbering
        normalized = re.sub(r'[，,]\s*', ', ', normalized)  # Normalize commas
        normalized = re.sub(r'\s*[,，]\s*', ', ', normalized)  # Clean comma spacing
        
        # Remove markdown formatting
        normalized = re.sub(r'\*\*([^*]+)\*\*', r'\1', normalized)  # Bold
        normalized = re.sub(r'\*([^*]+)\*', r'\1', normalized)  # Italic
        
        return normalized
    
    def extract_partial_results(self, response: str) -> PartialResult:
        """Extract partial results even from malformed responses"""
        parsed_items = []
        failed_items = []
        errors = []
        
        try:
            # Normalize first
            normalized = self.normalize_response_format(response)
            
            # Try to extract any valid classifications
            lines = normalized.split('\n')
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Try to parse this line
                    parsed = self._parse_single_line(line, i + 1)
                    if parsed:
                        parsed_items.append(self._format_classification(parsed))
                    else:
                        failed_items.append(line)
                except Exception as e:
                    failed_items.append(line)
                    errors.append(f"Line {i + 1}: {str(e)}")
            
            # Calculate confidence based on success rate
            total_items = len(parsed_items) + len(failed_items)
            confidence = len(parsed_items) / total_items if total_items > 0 else 0.0
            
            return PartialResult(
                parsed_items=parsed_items,
                failed_items=failed_items,
                confidence=confidence,
                parsing_method="partial_extraction",
                errors=errors
            )
            
        except Exception as e:
            logger.error(f"Failed to extract partial results: {e}")
            return PartialResult(
                parsed_items=[],
                failed_items=[response],
                confidence=0.0,
                parsing_method="failed",
                errors=[str(e)]
            )
    
    def generate_parsing_feedback(self, failures: List[str]) -> str:
        """Generate feedback for improving response format"""
        if not failures:
            return "No parsing failures detected."
        
        feedback_parts = [
            "응답 형식을 개선하기 위한 지침:",
            "",
            "1. 각 문장의 분류는 다음 형식을 정확히 따라주세요:",
            "   번호. 문장유형,감정극성,시제,확실성",
            "   예: 1. 사실형,긍정,현재,확실",
            "",
            "2. 유효한 분류 값들:",
            f"   - 문장유형: {', '.join(self.valid_types)}",
            f"   - 감정극성: {', '.join(self.valid_polarities)}",
            f"   - 시제: {', '.join(self.valid_tenses)}",
            f"   - 확실성: {', '.join(self.valid_certainties)}",
            "",
            "3. 각 줄은 하나의 분류만 포함해주세요.",
            "4. 번호는 1부터 순서대로 매겨주세요.",
            "5. 쉼표로 구분하고 공백은 최소화해주세요."
        ]
        
        # Add specific error analysis
        if len(failures) > 0:
            feedback_parts.extend([
                "",
                f"파싱 실패한 {len(failures)}개 항목 분석:",
            ])
            
            for i, failure in enumerate(failures[:3]):  # Show first 3 failures
                feedback_parts.append(f"   {i+1}. '{failure[:50]}...' - 형식 오류")
        
        return "\n".join(feedback_parts)
    
    def _parse_standard_format(self, response: str) -> Optional[List[str]]:
        """Parse standard numbered format: 1. type,polarity,tense,certainty"""
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Match pattern: number. classification
            match = re.match(r'^(\d+)\.?\s*(.+)$', line)
            if match:
                classification = match.group(2).strip()
                
                # Split by comma and validate
                parts = [p.strip() for p in classification.split(',')]
                if len(parts) == 4:
                    if self._validate_classification_parts(parts):
                        results.append(','.join(parts))
        
        return results if results else None
    
    def _parse_numbered_list(self, response: str) -> Optional[List[str]]:
        """Parse numbered list format with various separators"""
        results = []
        
        # Find all numbered items
        pattern = r'(\d+)\.?\s*([^\d\n]+?)(?=\d+\.|$)'
        matches = re.findall(pattern, response, re.DOTALL)
        
        for number, content in matches:
            content = content.strip()
            
            # Try to extract classification from content
            classification = self._extract_classification_from_text(content)
            if classification:
                results.append(classification)
        
        return results if results else None
    
    def _parse_json_format(self, response: str) -> Optional[List[str]]:
        """Parse JSON format responses"""
        try:
            # Try to find JSON in response
            json_match = re.search(r'\{.*\}|\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                
                results = []
                if isinstance(data, list):
                    for item in data:
                        classification = self._extract_from_json_item(item)
                        if classification:
                            results.append(classification)
                elif isinstance(data, dict):
                    for key, value in data.items():
                        classification = self._extract_from_json_item(value)
                        if classification:
                            results.append(classification)
                
                return results if results else None
        except json.JSONDecodeError:
            pass
        
        return None
    
    def _parse_comma_separated(self, response: str) -> Optional[List[str]]:
        """Parse comma-separated values format"""
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering if present
            line = re.sub(r'^\d+\.?\s*', '', line)
            
            # Split by comma
            parts = [p.strip() for p in line.split(',')]
            if len(parts) == 4:
                if self._validate_classification_parts(parts):
                    results.append(','.join(parts))
        
        return results if results else None
    
    def _parse_line_by_line(self, response: str) -> Optional[List[str]]:
        """Parse line-by-line format with flexible separators"""
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try various separators
            for separator in [',', '，', '|', ';', '\t']:
                parts = [p.strip() for p in line.split(separator)]
                if len(parts) == 4:
                    if self._validate_classification_parts(parts):
                        results.append(','.join(parts))
                        break
        
        return results if results else None
    
    def _parse_regex_extraction(self, response: str) -> Optional[List[str]]:
        """Parse using regex patterns to extract classifications"""
        results = []
        
        # Pattern to match Korean classification terms
        pattern = r'(사실형|추론형|대화형|예측형)[,，\s]*(긍정|부정|미정)[,，\s]*(과거|현재|미래)[,，\s]*(확실|불확실)'
        
        matches = re.findall(pattern, response)
        for match in matches:
            results.append(','.join(match))
        
        return results if results else None
    
    def _parse_fuzzy_matching(self, response: str) -> Optional[List[str]]:
        """Parse using fuzzy matching for classification terms"""
        results = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering
            line = re.sub(r'^\d+\.?\s*', '', line)
            
            # Try to find classification terms using fuzzy matching
            classification = self._fuzzy_match_classification(line)
            if classification:
                results.append(classification)
        
        return results if results else None
    
    def _parse_single_line(self, line: str, line_number: int) -> Optional[ParsedClassification]:
        """Parse a single line into classification"""
        # Remove numbering
        clean_line = re.sub(r'^\d+\.?\s*', '', line.strip())
        
        # Try comma-separated format first
        parts = [p.strip() for p in clean_line.split(',')]
        if len(parts) == 4:
            if self._validate_classification_parts(parts):
                return ParsedClassification(
                    index=line_number,
                    sentence_type=parts[0],
                    polarity=parts[1],
                    tense=parts[2],
                    certainty=parts[3],
                    raw_text=line
                )
        
        # Try fuzzy matching
        fuzzy_result = self._fuzzy_match_classification(clean_line)
        if fuzzy_result:
            parts = fuzzy_result.split(',')
            return ParsedClassification(
                index=line_number,
                sentence_type=parts[0],
                polarity=parts[1],
                tense=parts[2],
                certainty=parts[3],
                confidence=0.8,  # Lower confidence for fuzzy match
                raw_text=line
            )
        
        return None
    
    def _validate_classification_parts(self, parts: List[str]) -> bool:
        """Validate that classification parts are valid"""
        if len(parts) != 4:
            return False
        
        return (parts[0] in self.valid_types and
                parts[1] in self.valid_polarities and
                parts[2] in self.valid_tenses and
                parts[3] in self.valid_certainties)
    
    def _extract_classification_from_text(self, text: str) -> Optional[str]:
        """Extract classification from free text"""
        # Try comma-separated first
        parts = [p.strip() for p in text.split(',')]
        if len(parts) == 4 and self._validate_classification_parts(parts):
            return ','.join(parts)
        
        # Try fuzzy matching
        return self._fuzzy_match_classification(text)
    
    def _extract_from_json_item(self, item: Any) -> Optional[str]:
        """Extract classification from JSON item"""
        if isinstance(item, str):
            return self._extract_classification_from_text(item)
        elif isinstance(item, dict):
            # Try to extract from dict values
            values = []
            for key in ['type', 'polarity', 'tense', 'certainty']:
                if key in item:
                    values.append(str(item[key]))
            
            if len(values) == 4 and self._validate_classification_parts(values):
                return ','.join(values)
        elif isinstance(item, list) and len(item) == 4:
            str_values = [str(v) for v in item]
            if self._validate_classification_parts(str_values):
                return ','.join(str_values)
        
        return None
    
    def _fuzzy_match_classification(self, text: str) -> Optional[str]:
        """Perform fuzzy matching to extract classification"""
        # Find best matches for each category
        type_match = self._find_best_fuzzy_match(text, self.valid_types, self.type_alternatives)
        polarity_match = self._find_best_fuzzy_match(text, self.valid_polarities, self.polarity_alternatives)
        tense_match = self._find_best_fuzzy_match(text, self.valid_tenses, self.tense_alternatives)
        certainty_match = self._find_best_fuzzy_match(text, self.valid_certainties, self.certainty_alternatives)
        
        if all([type_match, polarity_match, tense_match, certainty_match]):
            return f"{type_match},{polarity_match},{tense_match},{certainty_match}"
        
        return None
    
    def _find_best_fuzzy_match(self, text: str, valid_values: List[str], alternatives: Dict[str, str]) -> Optional[str]:
        """Find best fuzzy match for a category"""
        text_lower = text.lower()
        
        # Exact match in valid values
        for value in valid_values:
            if value in text:
                return value
        
        # Match in alternatives
        for alt, canonical in alternatives.items():
            if alt in text_lower:
                return canonical
        
        # Partial match
        for value in valid_values:
            if any(char in text for char in value):
                char_overlap = sum(1 for char in value if char in text)
                if char_overlap >= len(value) * 0.6:  # 60% character overlap
                    return value
        
        return None
    
    def _format_classification(self, parsed: ParsedClassification) -> str:
        """Format parsed classification as string"""
        return f"{parsed.sentence_type},{parsed.polarity},{parsed.tense},{parsed.certainty}"
    
    def _parse_intelligent_recovery(self, response: str) -> Optional[List[str]]:
        """Intelligent recovery using machine learning-like pattern recognition"""
        results = []
        lines = response.strip().split('\n')
        
        # Build context from successful parses
        successful_patterns = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Try to identify patterns in the text
            words = line.split()
            
            # Look for Korean classification terms with context
            classification_candidates = []
            
            for i, word in enumerate(words):
                # Check if word contains classification terms
                for valid_type in self.valid_types:
                    if valid_type in word or any(char in word for char in valid_type):
                        classification_candidates.append((i, 'type', valid_type))
                
                for valid_polarity in self.valid_polarities:
                    if valid_polarity in word or any(char in word for char in valid_polarity):
                        classification_candidates.append((i, 'polarity', valid_polarity))
                
                for valid_tense in self.valid_tenses:
                    if valid_tense in word or any(char in word for char in valid_tense):
                        classification_candidates.append((i, 'tense', valid_tense))
                
                for valid_certainty in self.valid_certainties:
                    if valid_certainty in word or any(char in word for char in valid_certainty):
                        classification_candidates.append((i, 'certainty', valid_certainty))
            
            # Try to build complete classification from candidates
            if len(classification_candidates) >= 4:
                # Group by category
                by_category = {}
                for pos, category, value in classification_candidates:
                    if category not in by_category:
                        by_category[category] = []
                    by_category[category].append((pos, value))
                
                # Try to pick one from each category
                if all(cat in by_category for cat in ['type', 'polarity', 'tense', 'certainty']):
                    # Pick the first occurrence of each category
                    type_val = by_category['type'][0][1]
                    polarity_val = by_category['polarity'][0][1]
                    tense_val = by_category['tense'][0][1]
                    certainty_val = by_category['certainty'][0][1]
                    
                    classification = f"{type_val},{polarity_val},{tense_val},{certainty_val}"
                    if self._validate_classification_parts(classification.split(',')):
                        results.append(classification)
        
        return results if results else None
    
    def _parse_context_aware(self, response: str) -> Optional[List[str]]:
        """Context-aware parsing using surrounding text clues"""
        results = []
        lines = response.strip().split('\n')
        
        # Analyze the entire response for context clues
        full_text = response.lower()
        
        # Detect if this is a classification task response
        classification_indicators = [
            '분류', '유형', '극성', '시제', '확실성',
            'classification', 'type', 'polarity', 'tense', 'certainty'
        ]
        
        has_classification_context = any(indicator in full_text for indicator in classification_indicators)
        
        if not has_classification_context:
            return None
        
        # Try to extract classifications with context awareness
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes and suffixes
            clean_line = line
            
            # Remove numbering and common prefixes
            clean_line = re.sub(r'^\d+[\.\)]\s*', '', clean_line)
            clean_line = re.sub(r'^[-*]\s*', '', clean_line)
            clean_line = re.sub(r'^답:\s*', '', clean_line)
            clean_line = re.sub(r'^결과:\s*', '', clean_line)
            
            # Try to find classification pattern with flexible separators
            # Look for 4 Korean terms that match our categories
            korean_terms = re.findall(r'[가-힣]+', clean_line)
            
            if len(korean_terms) >= 4:
                # Try different combinations of 4 consecutive terms
                for start_idx in range(len(korean_terms) - 3):
                    candidate_terms = korean_terms[start_idx:start_idx + 4]
                    
                    # Check if these terms match our classification categories
                    matched_terms = []
                    
                    for term in candidate_terms:
                        best_match = None
                        best_category = None
                        
                        # Find best match in each category
                        for category, valid_values in [
                            ('type', self.valid_types),
                            ('polarity', self.valid_polarities),
                            ('tense', self.valid_tenses),
                            ('certainty', self.valid_certainties)
                        ]:
                            for valid_value in valid_values:
                                if term == valid_value or term in valid_value or valid_value in term:
                                    if len(term) > len(best_match or ''):
                                        best_match = valid_value
                                        best_category = category
                        
                        if best_match:
                            matched_terms.append((best_category, best_match))
                    
                    # Check if we have one term from each category
                    categories_found = set(cat for cat, _ in matched_terms)
                    if len(categories_found) == 4:
                        # Arrange in correct order
                        term_by_category = {cat: term for cat, term in matched_terms}
                        
                        classification = f"{term_by_category['type']},{term_by_category['polarity']},{term_by_category['tense']},{term_by_category['certainty']}"
                        
                        if self._validate_classification_parts(classification.split(',')):
                            results.append(classification)
                            break
        
        return results if results else None
    
    def enhance_response_format_recovery(self, response: str, error_context: Optional[str] = None) -> str:
        """Enhanced response format recovery with error context"""
        if not response:
            return ""
        
        # Start with basic normalization
        enhanced = self.normalize_response_format(response)
        
        # Apply error-context specific fixes
        if error_context:
            if "parsing" in error_context.lower():
                # Focus on format standardization
                enhanced = self._fix_parsing_issues(enhanced)
            elif "incomplete" in error_context.lower():
                # Try to complete partial responses
                enhanced = self._complete_partial_response(enhanced)
            elif "format" in error_context.lower():
                # Standardize format more aggressively
                enhanced = self._standardize_format_aggressively(enhanced)
        
        # Apply intelligent corrections
        enhanced = self._apply_intelligent_corrections(enhanced)
        
        return enhanced
    
    def _fix_parsing_issues(self, response: str) -> str:
        """Fix common parsing issues"""
        # Fix missing commas between classifications
        response = re.sub(r'([가-힣]+)([가-힣]+)([가-힣]+)([가-힣]+)', r'\1,\2,\3,\4', response)
        
        # Fix spacing issues around commas
        response = re.sub(r'\s*,\s*', ',', response)
        
        # Fix numbering format
        response = re.sub(r'(\d+)[\.\)]\s*', r'\1. ', response)
        
        return response
    
    def _complete_partial_response(self, response: str) -> str:
        """Try to complete partial responses"""
        lines = response.strip().split('\n')
        completed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Check if line looks incomplete (less than 4 classifications)
            parts = [p.strip() for p in line.split(',')]
            
            if len(parts) < 4 and len(parts) > 0:
                # Try to infer missing parts based on context
                while len(parts) < 4:
                    if len(parts) == 1:  # Only type given
                        parts.append("미정")  # Default polarity
                    elif len(parts) == 2:  # Type and polarity given
                        parts.append("현재")  # Default tense
                    elif len(parts) == 3:  # Type, polarity, tense given
                        parts.append("확실")  # Default certainty
                
                # Validate the completed classification
                if self._validate_classification_parts(parts):
                    # Reconstruct the line with numbering
                    line_number = len(completed_lines) + 1
                    completed_line = f"{line_number}. {','.join(parts)}"
                    completed_lines.append(completed_line)
                else:
                    completed_lines.append(line)
            else:
                completed_lines.append(line)
        
        return '\n'.join(completed_lines)
    
    def _standardize_format_aggressively(self, response: str) -> str:
        """Aggressively standardize response format"""
        lines = response.strip().split('\n')
        standardized_lines = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Extract any Korean classification terms
            korean_terms = re.findall(r'[가-힣]+', line)
            
            # Filter to only valid classification terms
            valid_terms = []
            for term in korean_terms:
                if (term in self.valid_types or 
                    term in self.valid_polarities or 
                    term in self.valid_tenses or 
                    term in self.valid_certainties):
                    valid_terms.append(term)
            
            # If we have exactly 4 valid terms, format them properly
            if len(valid_terms) == 4:
                standardized_line = f"{i + 1}. {','.join(valid_terms)}"
                standardized_lines.append(standardized_line)
            elif len(valid_terms) > 4:
                # Take first 4 valid terms
                standardized_line = f"{i + 1}. {','.join(valid_terms[:4])}"
                standardized_lines.append(standardized_line)
            else:
                # Keep original line if we can't standardize
                standardized_lines.append(line)
        
        return '\n'.join(standardized_lines)
    
    def _apply_intelligent_corrections(self, response: str) -> str:
        """Apply intelligent corrections based on common error patterns"""
        # Fix common Korean input method errors
        corrections = {
            '긍적': '긍정',
            '부적': '부정',
            '확신': '확실',
            '불확신': '불확실',
            '사실': '사실형',
            '추론': '추론형',
            '대화': '대화형',
            '예측': '예측형'
        }
        
        for wrong, correct in corrections.items():
            response = response.replace(wrong, correct)
        
        # Fix English alternatives
        english_corrections = {
            'fact': '사실형',
            'inference': '추론형',
            'dialogue': '대화형',
            'prediction': '예측형',
            'positive': '긍정',
            'negative': '부정',
            'neutral': '미정',
            'past': '과거',
            'present': '현재',
            'future': '미래',
            'certain': '확실',
            'uncertain': '불확실'
        }
        
        for english, korean in english_corrections.items():
            response = re.sub(r'\b' + english + r'\b', korean, response, flags=re.IGNORECASE)
        
        return response