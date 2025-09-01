"""
Result analysis service for classification results
"""
import re
from typing import List, Dict, Any, Tuple
import logging
from models.data_models import Sample, ErrorCase, AnalysisReport
from models.exceptions import ValidationError

logger = logging.getLogger("gemini_optimizer.result_analyzer")

class ResultAnalyzer:
    """Analyzer for classification results and error patterns"""
    
    def __init__(self):
        self.valid_types = ["사실형", "추론형", "대화형", "예측형"]
        self.valid_polarities = ["긍정", "부정", "미정"]
        self.valid_tenses = ["과거", "현재", "미래"]
        self.valid_certainties = ["확실", "불확실"]
        
        logger.info("ResultAnalyzer initialized")
    
    def compare_results(self, predictions: List[str], correct_answers: List[str], 
                       samples: List[Sample] = None) -> Dict[str, Any]:
        """Compare predictions with correct answers and return detailed analysis"""
        
        if len(predictions) != len(correct_answers):
            logger.warning(f"Prediction count ({len(predictions)}) != answer count ({len(correct_answers)})")
            min_length = min(len(predictions), len(correct_answers))
            predictions = predictions[:min_length]
            correct_answers = correct_answers[:min_length]
            if samples:
                samples = samples[:min_length]
        
        if not predictions:
            raise ValidationError("No predictions to analyze")
        
        logger.info(f"Analyzing {len(predictions)} predictions")
        
        analysis_result = {
            "total_samples": len(predictions),
            "correct_predictions": 0,
            "overall_accuracy": 0.0,
            "category_accuracy": {
                "type": {"correct": 0, "total": 0, "accuracy": 0.0},
                "polarity": {"correct": 0, "total": 0, "accuracy": 0.0},
                "tense": {"correct": 0, "total": 0, "accuracy": 0.0},
                "certainty": {"correct": 0, "total": 0, "accuracy": 0.0}
            },
            "error_details": [],
            "error_patterns": {},
            "category_confusion_matrix": {},
            "statistics": {}
        }
        
        correct_count = 0
        category_correct = {"type": 0, "polarity": 0, "tense": 0, "certainty": 0}
        category_total = {"type": 0, "polarity": 0, "tense": 0, "certainty": 0}
        
        # Initialize confusion matrices
        confusion_matrices = {
            "type": self._init_confusion_matrix(self.valid_types),
            "polarity": self._init_confusion_matrix(self.valid_polarities),
            "tense": self._init_confusion_matrix(self.valid_tenses),
            "certainty": self._init_confusion_matrix(self.valid_certainties)
        }
        
        # Define categories at the beginning
        categories = ["type", "polarity", "tense", "certainty"]
        
        for i, (pred, answer) in enumerate(zip(predictions, correct_answers)):
            pred_classification = self._extract_classification(pred)
            answer_classification = self._extract_classification(answer)
            
            # Parse classifications
            pred_parts = self._parse_classification(pred_classification)
            answer_parts = self._parse_classification(answer_classification)
            
            if not pred_parts or not answer_parts:
                logger.warning(f"Failed to parse classification at index {i}")
                # Still count this as an error and continue with category analysis
                error_detail = {
                    "index": i + 1,
                    "sentence": samples[i].sentence if samples and i < len(samples) else f"Sample {i+1}",
                    "predicted": pred_classification,
                    "expected": answer_classification,
                    "error_categories": ["format_error"]
                }
                analysis_result["error_details"].append(error_detail)
                
                # Skip category analysis for this item but continue with others
                for category in categories:
                    category_total[category] += 1
                continue
            
            # Overall accuracy
            is_correct = pred_classification == answer_classification
            if is_correct:
                correct_count += 1
            else:
                # Record error details
                error_detail = {
                    "index": i + 1,
                    "sentence": samples[i].sentence if samples and i < len(samples) else f"Sample {i+1}",
                    "predicted": pred_classification,
                    "expected": answer_classification,
                    "error_categories": []
                }
                analysis_result["error_details"].append(error_detail)
            
            # Category-wise analysis
            for j, category in enumerate(categories):
                category_total[category] += 1
                
                pred_val = pred_parts[j]
                answer_val = answer_parts[j]
                
                # Update confusion matrix
                if category in confusion_matrices:
                    if answer_val in confusion_matrices[category] and pred_val in confusion_matrices[category][answer_val]:
                        confusion_matrices[category][answer_val][pred_val] += 1
                
                if pred_val == answer_val:
                    category_correct[category] += 1
                else:
                    # Record category error
                    if not is_correct:  # Only add to error categories if overall prediction was wrong
                        error_detail["error_categories"].append({
                            "category": category,
                            "expected": answer_val,
                            "predicted": pred_val
                        })
                    
                    # Update error patterns
                    pattern_key = f"{category}: {answer_val} -> {pred_val}"
                    analysis_result["error_patterns"][pattern_key] = analysis_result["error_patterns"].get(pattern_key, 0) + 1
        
        # Calculate final accuracies
        analysis_result["correct_predictions"] = correct_count
        analysis_result["overall_accuracy"] = correct_count / len(predictions) if predictions else 0.0
        
        for category in categories:
            total = category_total[category]
            correct = category_correct[category]
            accuracy = correct / total if total > 0 else 0.0
            
            analysis_result["category_accuracy"][category] = {
                "correct": correct,
                "total": total,
                "accuracy": accuracy
            }
        
        analysis_result["category_confusion_matrix"] = confusion_matrices
        analysis_result["statistics"] = self._calculate_statistics(analysis_result)
        
        logger.info(f"Analysis complete: {correct_count}/{len(predictions)} correct ({analysis_result['overall_accuracy']:.4f})")
        
        return analysis_result
    
    def extract_errors(self, analysis_result: Dict[str, Any]) -> List[ErrorCase]:
        """Extract error cases from analysis result"""
        errors = []
        
        for error_detail in analysis_result.get("error_details", []):
            error_types = []
            for error_cat in error_detail.get("error_categories", []):
                error_types.append(error_cat["category"])
            
            error_case = ErrorCase(
                question_id=error_detail["index"],
                sentence=error_detail["sentence"],
                expected=error_detail["expected"],
                predicted=error_detail["predicted"],
                error_type=",".join(error_types) if error_types else "unknown"
            )
            errors.append(error_case)
        
        logger.info(f"Extracted {len(errors)} error cases")
        return errors
    
    def generate_error_report(self, errors: List[ErrorCase]) -> str:
        """Generate detailed error report"""
        if not errors:
            return "No errors found in classification results."
        
        report_lines = [
            "=== Classification Error Report ===",
            f"Total Errors: {len(errors)}",
            "",
            "Error Details:",
            "-" * 50
        ]
        
        # Group errors by type
        error_by_type = {}
        for error in errors:
            error_types = error.error_type.split(",")
            for error_type in error_types:
                if error_type not in error_by_type:
                    error_by_type[error_type] = []
                error_by_type[error_type].append(error)
        
        for error_type, error_list in error_by_type.items():
            report_lines.append(f"\n{error_type.upper()} Errors ({len(error_list)}):")
            report_lines.append("-" * 30)
            
            for error in error_list[:10]:  # Show first 10 errors of each type
                report_lines.append(f"ID {error.question_id}: {error.sentence[:100]}...")
                report_lines.append(f"  Expected: {error.expected}")
                report_lines.append(f"  Predicted: {error.predicted}")
                report_lines.append("")
            
            if len(error_list) > 10:
                report_lines.append(f"... and {len(error_list) - 10} more {error_type} errors")
                report_lines.append("")
        
        # Error pattern summary
        report_lines.extend([
            "\n=== Error Pattern Summary ===",
            f"Most common error types: {', '.join(list(error_by_type.keys())[:5])}",
            f"Average errors per sample: {len(errors) / len(set(e.question_id for e in errors)):.2f}"
        ])
        
        return "\n".join(report_lines)
    
    def _extract_classification(self, response: str) -> str:
        """Extract classification part from response (remove number prefix)"""
        match = re.match(r'^\d+\.\s*(.+)$', response.strip())
        if match:
            return match.group(1).strip()
        return response.strip()
    
    def _parse_classification(self, classification: str) -> List[str]:
        """Parse classification string into components"""
        parts = classification.split(',')
        if len(parts) != 4:
            return None
        return [part.strip() for part in parts]
    
    def _init_confusion_matrix(self, valid_values: List[str]) -> Dict[str, Dict[str, int]]:
        """Initialize confusion matrix for a category"""
        matrix = {}
        for true_val in valid_values:
            matrix[true_val] = {}
            for pred_val in valid_values:
                matrix[true_val][pred_val] = 0
        return matrix
    
    def _calculate_statistics(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate additional statistics"""
        stats = {
            "error_rate": 1.0 - analysis_result["overall_accuracy"],
            "total_errors": len(analysis_result["error_details"]),
            "most_common_errors": [],
            "category_performance": {}
        }
        
        # Most common error patterns
        error_patterns = analysis_result["error_patterns"]
        if error_patterns:
            sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
            stats["most_common_errors"] = sorted_patterns[:5]
        
        # Category performance ranking
        category_acc = analysis_result["category_accuracy"]
        category_ranking = sorted(category_acc.items(), key=lambda x: x[1]["accuracy"], reverse=True)
        
        for i, (category, acc_info) in enumerate(category_ranking):
            stats["category_performance"][category] = {
                "rank": i + 1,
                "accuracy": acc_info["accuracy"],
                "error_count": acc_info["total"] - acc_info["correct"]
            }
        
        return stats
    
    def get_improvement_suggestions(self, analysis_result: Dict[str, Any]) -> List[str]:
        """Generate improvement suggestions based on analysis"""
        suggestions = []
        
        # Overall accuracy suggestions
        overall_acc = analysis_result["overall_accuracy"]
        if overall_acc < 0.7:
            suggestions.append("전체 정확도가 낮습니다. 시스템 프롬프트의 분류 기준을 더 명확하게 정의해야 합니다.")
        elif overall_acc < 0.9:
            suggestions.append("정확도 개선이 필요합니다. 오류 패턴을 분석하여 특정 분류 기준을 강화하세요.")
        
        # Category-specific suggestions
        category_acc = analysis_result["category_accuracy"]
        
        for category, acc_info in category_acc.items():
            if acc_info["accuracy"] < 0.8:
                category_names = {
                    "type": "유형 분류",
                    "polarity": "극성 분류", 
                    "tense": "시제 분류",
                    "certainty": "확실성 분류"
                }
                suggestions.append(f"{category_names.get(category, category)} 정확도가 낮습니다 ({acc_info['accuracy']:.2f}). 해당 분류 기준을 재검토하세요.")
        
        # Error pattern suggestions
        error_patterns = analysis_result["error_patterns"]
        if error_patterns:
            most_common = max(error_patterns.items(), key=lambda x: x[1])
            suggestions.append(f"가장 흔한 오류: '{most_common[0]}' ({most_common[1]}회). 이 패턴에 대한 명확한 지침을 추가하세요.")
        
        # Specific improvement areas
        stats = analysis_result.get("statistics", {})
        category_performance = stats.get("category_performance", {})
        
        worst_category = None
        worst_accuracy = 1.0
        for category, perf in category_performance.items():
            if perf["accuracy"] < worst_accuracy:
                worst_accuracy = perf["accuracy"]
                worst_category = category
        
        if worst_category and worst_accuracy < 0.85:
            category_names = {
                "type": "유형 분류",
                "polarity": "극성 분류",
                "tense": "시제 분류", 
                "certainty": "확실성 분류"
            }
            suggestions.append(f"{category_names.get(worst_category, worst_category)}가 가장 취약한 영역입니다. 더 많은 예시와 명확한 기준을 제공하세요.")
        
        return suggestions if suggestions else ["현재 성능이 양호합니다. 미세 조정을 통해 더 개선할 수 있습니다."]