"""
Gemini Flash classifier for Korean sentence classification
"""
import re
from typing import List, Dict, Any, Tuple
import logging
from services.gemini_client import GeminiClient
from models.data_models import Sample
from models.exceptions import APIError, ValidationError, InvalidResponseError
from config import OptimizationConfig

logger = logging.getLogger("gemini_optimizer.flash_classifier")

class GeminiFlashClassifier:
    """Gemini Flash-based sentence classifier"""
    
    def __init__(self, config: OptimizationConfig, system_prompt: str):
        self.config = config
        self.system_prompt = system_prompt
        self.client = GeminiClient(config)
        self.model = self.client.get_flash_model()
        
        # Classification validation patterns
        self.valid_types = ["사실형", "추론형", "대화형", "예측형"]
        self.valid_polarities = ["긍정", "부정", "미정"]
        self.valid_tenses = ["과거", "현재", "미래"]
        self.valid_certainties = ["확실", "불확실"]
        
        logger.info("GeminiFlashClassifier initialized")
    
    def classify_single(self, question: str) -> str:
        """Classify a single question"""
        try:
            # Prepare the full prompt
            full_prompt = f"{self.system_prompt}\n\n{question}"
            
            # Generate response
            response = self.client.generate_content_with_retry(self.model, full_prompt)
            
            # Parse and validate response
            parsed_response = self._parse_response(response, 1)
            
            if not parsed_response:
                raise InvalidResponseError(f"Failed to parse response: {response}")
            
            return parsed_response[0]
            
        except Exception as e:
            logger.error(f"Failed to classify single question: {e}")
            raise APIError(f"Classification failed: {e}")
    
    def classify_batch(self, questions: List[str]) -> List[str]:
        """Classify a batch of questions with performance optimization"""
        try:
            logger.info(f"Classifying batch of {len(questions)} questions")
            
            # Use batch processor for large batches
            if len(questions) > self.config.batch_size:
                from utils.performance_optimizer import BatchProcessor
                processor = BatchProcessor(
                    batch_size=self.config.batch_size,
                    max_workers=2,  # Conservative for API limits
                    delay_between_batches=1.0
                )
                
                def classify_sub_batch(batch_text: str) -> str:
                    full_prompt = f"{self.system_prompt}\n\n{batch_text}"
                    return self.client.generate_content_with_retry(self.model, full_prompt)
                
                return processor.process_batches(questions, classify_sub_batch, use_cache=True)
            
            # Standard batch processing for smaller batches
            questions_text = "\n".join(questions)
            full_prompt = f"{self.system_prompt}\n\n{questions_text}"
            
            # Generate response
            response = self.client.generate_content_with_retry(self.model, full_prompt)
            
            # Parse and validate response
            parsed_responses = self._parse_response(response, len(questions))
            
            if len(parsed_responses) != len(questions):
                logger.warning(f"Expected {len(questions)} responses, got {len(parsed_responses)}")
                # Try to handle partial responses
                parsed_responses = self._handle_partial_response(parsed_responses, len(questions))
            
            logger.info(f"Successfully classified {len(parsed_responses)} questions")
            return parsed_responses
            
        except Exception as e:
            logger.error(f"Failed to classify batch: {e}")
            raise APIError(f"Batch classification failed: {e}")
    
    def _parse_response(self, response: str, expected_count: int) -> List[str]:
        """Parse API response and extract classifications"""
        try:
            lines = response.strip().split('\n')
            parsed_responses = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Look for pattern: "number. type,polarity,tense,certainty"
                match = re.match(r'^(\d+)\.\s*(.+)$', line)
                if match:
                    number = int(match.group(1))
                    classification = match.group(2).strip()
                    
                    # Validate classification format
                    if self._validate_classification_format(classification):
                        parsed_responses.append(f"{number}. {classification}")
                    else:
                        logger.warning(f"Invalid classification format: {classification}")
                        # Try to fix common issues
                        fixed_classification = self._fix_classification_format(classification)
                        if fixed_classification:
                            parsed_responses.append(f"{number}. {fixed_classification}")
                        else:
                            logger.error(f"Could not fix classification: {classification}")
            
            return parsed_responses
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return []
    
    def _validate_classification_format(self, classification: str) -> bool:
        """Validate classification format: type,polarity,tense,certainty"""
        parts = classification.split(',')
        
        if len(parts) != 4:
            return False
        
        type_val, polarity_val, tense_val, certainty_val = [part.strip() for part in parts]
        
        return (
            type_val in self.valid_types and
            polarity_val in self.valid_polarities and
            tense_val in self.valid_tenses and
            certainty_val in self.valid_certainties
        )
    
    def _fix_classification_format(self, classification: str) -> str:
        """Try to fix common classification format issues"""
        try:
            # Remove extra spaces and normalize
            classification = re.sub(r'\s+', ' ', classification.strip())
            
            # Split by comma or other separators
            parts = re.split(r'[,，\s]+', classification)
            
            if len(parts) < 4:
                return None
            
            # Take first 4 parts
            parts = parts[:4]
            
            # Try to match each part to valid values
            fixed_parts = []
            
            # Type matching
            type_val = self._find_closest_match(parts[0], self.valid_types)
            if not type_val:
                return None
            fixed_parts.append(type_val)
            
            # Polarity matching
            polarity_val = self._find_closest_match(parts[1], self.valid_polarities)
            if not polarity_val:
                return None
            fixed_parts.append(polarity_val)
            
            # Tense matching
            tense_val = self._find_closest_match(parts[2], self.valid_tenses)
            if not tense_val:
                return None
            fixed_parts.append(tense_val)
            
            # Certainty matching
            certainty_val = self._find_closest_match(parts[3], self.valid_certainties)
            if not certainty_val:
                return None
            fixed_parts.append(certainty_val)
            
            return ','.join(fixed_parts)
            
        except Exception as e:
            logger.debug(f"Failed to fix classification format: {e}")
            return None
    
    def _find_closest_match(self, value: str, valid_values: List[str]) -> str:
        """Find closest match from valid values"""
        value = value.strip()
        
        # Exact match
        if value in valid_values:
            return value
        
        # Handle common mismatches
        value_mapping = {
            "중립": "긍정",  # Map 중립 to 긍정
            "neutral": "긍정",
            "positive": "긍정",
            "negative": "부정",
            "uncertain": "미정",
            "past": "과거",
            "present": "현재", 
            "future": "미래",
            "certain": "확실",
            "uncertain": "불확실"
        }
        
        if value in value_mapping:
            return value_mapping[value]
        
        # Partial match
        for valid_val in valid_values:
            if value in valid_val or valid_val in value:
                return valid_val
        
        # If no match found, return the first valid value as default
        return valid_values[0] if valid_values else None
    
    def _handle_partial_response(self, responses: List[str], expected_count: int) -> List[str]:
        """Handle partial responses by padding with default values"""
        if len(responses) >= expected_count:
            return responses[:expected_count]
        
        # Pad with default classifications for missing responses
        default_classification = "사실형,긍정,현재,확실"
        
        while len(responses) < expected_count:
            response_num = len(responses) + 1
            responses.append(f"{response_num}. {default_classification}")
            logger.warning(f"Added default classification for missing response {response_num}")
        
        return responses
    
    def calculate_accuracy(self, predictions: List[str], correct_answers: List[str]) -> float:
        """Calculate classification accuracy"""
        if len(predictions) != len(correct_answers):
            logger.warning(f"Prediction count ({len(predictions)}) != answer count ({len(correct_answers)})")
            min_length = min(len(predictions), len(correct_answers))
            predictions = predictions[:min_length]
            correct_answers = correct_answers[:min_length]
        
        if not predictions:
            return 0.0
        
        correct_count = 0
        for pred, answer in zip(predictions, correct_answers):
            # Extract classification parts (remove number prefix)
            pred_classification = self._extract_classification(pred)
            answer_classification = self._extract_classification(answer)
            
            if pred_classification == answer_classification:
                correct_count += 1
        
        accuracy = correct_count / len(predictions)
        logger.info(f"Accuracy: {correct_count}/{len(predictions)} = {accuracy:.4f}")
        
        return accuracy
    
    def _extract_classification(self, response: str) -> str:
        """Extract classification part from response (remove number prefix)"""
        match = re.match(r'^\d+\.\s*(.+)$', response.strip())
        if match:
            return match.group(1).strip()
        return response.strip()
    
    def get_detailed_results(self, predictions: List[str], correct_answers: List[str]) -> Dict[str, Any]:
        """Get detailed classification results"""
        if len(predictions) != len(correct_answers):
            min_length = min(len(predictions), len(correct_answers))
            predictions = predictions[:min_length]
            correct_answers = correct_answers[:min_length]
        
        results = {
            "total_samples": len(predictions),
            "correct_predictions": 0,
            "accuracy": 0.0,
            "category_accuracy": {
                "type": 0,
                "polarity": 0,
                "tense": 0,
                "certainty": 0
            },
            "errors": [],
            "error_patterns": {}
        }
        
        if not predictions:
            return results
        
        correct_count = 0
        category_correct = {"type": 0, "polarity": 0, "tense": 0, "certainty": 0}
        
        for i, (pred, answer) in enumerate(zip(predictions, correct_answers)):
            pred_classification = self._extract_classification(pred)
            answer_classification = self._extract_classification(answer)
            
            # Overall accuracy
            if pred_classification == answer_classification:
                correct_count += 1
            else:
                # Record error
                error_info = {
                    "index": i + 1,
                    "predicted": pred_classification,
                    "expected": answer_classification
                }
                results["errors"].append(error_info)
            
            # Category-wise accuracy
            pred_parts = pred_classification.split(',')
            answer_parts = answer_classification.split(',')
            
            if len(pred_parts) == 4 and len(answer_parts) == 4:
                categories = ["type", "polarity", "tense", "certainty"]
                for j, category in enumerate(categories):
                    if pred_parts[j].strip() == answer_parts[j].strip():
                        category_correct[category] += 1
        
        # Calculate accuracies
        results["correct_predictions"] = correct_count
        results["accuracy"] = correct_count / len(predictions)
        
        for category in category_correct:
            results["category_accuracy"][category] = category_correct[category] / len(predictions)
        
        # Analyze error patterns
        error_patterns = {}
        for error in results["errors"]:
            pred_parts = error["predicted"].split(',')
            expected_parts = error["expected"].split(',')
            
            if len(pred_parts) == 4 and len(expected_parts) == 4:
                categories = ["type", "polarity", "tense", "certainty"]
                for j, category in enumerate(categories):
                    if pred_parts[j].strip() != expected_parts[j].strip():
                        pattern_key = f"{category}: {expected_parts[j].strip()} -> {pred_parts[j].strip()}"
                        error_patterns[pattern_key] = error_patterns.get(pattern_key, 0) + 1
        
        results["error_patterns"] = error_patterns
        
        return results
    
    def update_system_prompt(self, new_prompt: str) -> None:
        """Update system prompt"""
        self.system_prompt = new_prompt
        logger.info("System prompt updated")
    
    def get_system_prompt(self) -> str:
        """Get current system prompt"""
        return self.system_prompt