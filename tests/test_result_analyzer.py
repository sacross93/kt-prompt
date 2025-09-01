"""
Unit tests for result analyzer
"""
import unittest
from services.result_analyzer import ResultAnalyzer
from models.data_models import Sample, ErrorCase

class TestResultAnalyzer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = ResultAnalyzer()
        
        self.samples = [
            Sample(1, "테스트 문장 1", "사실형", "긍정", "현재", "확실"),
            Sample(2, "테스트 문장 2", "추론형", "부정", "과거", "불확실"),
            Sample(3, "테스트 문장 3", "대화형", "미정", "미래", "확실")
        ]
        
        self.correct_answers = [
            "1. 사실형,긍정,현재,확실",
            "2. 추론형,부정,과거,불확실", 
            "3. 대화형,미정,미래,확실"
        ]
    
    def test_compare_results_perfect_match(self):
        """Test comparison with perfect predictions"""
        predictions = self.correct_answers.copy()
        
        result = self.analyzer.compare_results(predictions, self.correct_answers, self.samples)
        
        self.assertEqual(result["overall_accuracy"], 1.0)
        self.assertEqual(result["correct_predictions"], 3)
        self.assertEqual(len(result["error_details"]), 0)
    
    def test_compare_results_with_errors(self):
        """Test comparison with some errors"""
        predictions = [
            "1. 추론형,긍정,현재,확실",  # Wrong type
            "2. 추론형,부정,과거,불확실",  # Correct
            "3. 대화형,긍정,미래,확실"   # Wrong polarity
        ]
        
        result = self.analyzer.compare_results(predictions, self.correct_answers, self.samples)
        
        self.assertEqual(result["overall_accuracy"], 1/3)  # Only 1 correct out of 3
        self.assertEqual(result["correct_predictions"], 1)
        self.assertEqual(len(result["error_details"]), 2)
        
        # Check category accuracies
        self.assertEqual(result["category_accuracy"]["type"]["correct"], 2)  # 2 correct types
        self.assertEqual(result["category_accuracy"]["polarity"]["correct"], 2)  # 2 correct polarities
    
    def test_extract_errors(self):
        """Test error extraction"""
        predictions = [
            "1. 추론형,긍정,현재,확실",  # Wrong type
            "2. 추론형,부정,과거,불확실",  # Correct
            "3. 대화형,긍정,미래,확실"   # Wrong polarity
        ]
        
        result = self.analyzer.compare_results(predictions, self.correct_answers, self.samples)
        errors = self.analyzer.extract_errors(result)
        
        self.assertEqual(len(errors), 2)
        self.assertIsInstance(errors[0], ErrorCase)
        self.assertEqual(errors[0].question_id, 1)
        self.assertIn("type", errors[0].error_type)
    
    def test_generate_error_report(self):
        """Test error report generation"""
        errors = [
            ErrorCase(1, "테스트 문장", "사실형,긍정,현재,확실", "추론형,긍정,현재,확실", "type"),
            ErrorCase(2, "테스트 문장 2", "대화형,미정,미래,확실", "대화형,긍정,미래,확실", "polarity")
        ]
        
        report = self.analyzer.generate_error_report(errors)
        
        self.assertIn("Total Errors: 2", report)
        self.assertIn("TYPE Errors", report)
        self.assertIn("POLARITY Errors", report)
    
    def test_extract_classification(self):
        """Test classification extraction from response"""
        response = "1. 사실형,긍정,현재,확실"
        classification = self.analyzer._extract_classification(response)
        
        self.assertEqual(classification, "사실형,긍정,현재,확실")
    
    def test_parse_classification(self):
        """Test classification parsing"""
        classification = "사실형,긍정,현재,확실"
        parts = self.analyzer._parse_classification(classification)
        
        self.assertEqual(len(parts), 4)
        self.assertEqual(parts[0], "사실형")
        self.assertEqual(parts[1], "긍정")
        self.assertEqual(parts[2], "현재")
        self.assertEqual(parts[3], "확실")
    
    def test_parse_classification_invalid(self):
        """Test parsing invalid classification"""
        classification = "사실형,긍정"  # Only 2 parts instead of 4
        parts = self.analyzer._parse_classification(classification)
        
        self.assertIsNone(parts)
    
    def test_get_improvement_suggestions(self):
        """Test improvement suggestions generation"""
        # Create analysis result with low accuracy
        result = {
            "overall_accuracy": 0.6,
            "category_accuracy": {
                "type": {"accuracy": 0.5},
                "polarity": {"accuracy": 0.8},
                "tense": {"accuracy": 0.7},
                "certainty": {"accuracy": 0.9}
            },
            "error_patterns": {
                "type: 사실형 -> 추론형": 5,
                "polarity: 긍정 -> 부정": 2
            },
            "statistics": {
                "category_performance": {
                    "type": {"accuracy": 0.5},
                    "polarity": {"accuracy": 0.8},
                    "tense": {"accuracy": 0.7},
                    "certainty": {"accuracy": 0.9}
                }
            }
        }
        
        suggestions = self.analyzer.get_improvement_suggestions(result)
        
        self.assertGreater(len(suggestions), 0)
        self.assertTrue(any("정확도" in s for s in suggestions))
    
    def test_init_confusion_matrix(self):
        """Test confusion matrix initialization"""
        valid_values = ["A", "B", "C"]
        matrix = self.analyzer._init_confusion_matrix(valid_values)
        
        self.assertEqual(len(matrix), 3)
        self.assertEqual(len(matrix["A"]), 3)
        self.assertEqual(matrix["A"]["B"], 0)
    
    def test_calculate_statistics(self):
        """Test statistics calculation"""
        analysis_result = {
            "overall_accuracy": 0.8,
            "error_details": [{"error": "test"}] * 5,
            "error_patterns": {"pattern1": 3, "pattern2": 2},
            "category_accuracy": {
                "type": {"accuracy": 0.9, "total": 10, "correct": 9},
                "polarity": {"accuracy": 0.7, "total": 10, "correct": 7}
            }
        }
        
        stats = self.analyzer._calculate_statistics(analysis_result)
        
        self.assertEqual(stats["error_rate"], 0.2)
        self.assertEqual(stats["total_errors"], 5)
        self.assertIn("most_common_errors", stats)
        self.assertIn("category_performance", stats)

if __name__ == '__main__':
    unittest.main()