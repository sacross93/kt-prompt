"""
Integration tests for Gemini Prompt Optimizer
"""
import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

class TestGeminiOptimizerIntegration(unittest.TestCase):
    """Integration tests for the complete optimization system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.prompt_dir = os.path.join(self.temp_dir, "prompts")
        self.analysis_dir = os.path.join(self.temp_dir, "analysis")
        
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Create test CSV data
        self.test_data = [
            {"id": 1, "sentence": "오늘은 날씨가 좋다.", "type": "사실형", "polarity": "긍정", "tense": "현재", "certainty": "확실"},
            {"id": 2, "sentence": "내일 비가 올 것 같다.", "type": "예측형", "polarity": "미정", "tense": "미래", "certainty": "불확실"},
            {"id": 3, "sentence": "어제 영화를 봤다.", "type": "사실형", "polarity": "긍정", "tense": "과거", "certainty": "확실"},
            {"id": 4, "sentence": "이 영화는 재미없을 것이다.", "type": "예측형", "polarity": "부정", "tense": "미래", "certainty": "불확실"},
            {"id": 5, "sentence": "안녕하세요, 반갑습니다.", "type": "대화형", "polarity": "긍정", "tense": "현재", "certainty": "확실"}
        ]
        
        # Create CSV file
        self.csv_file = os.path.join(self.temp_dir, "test_samples.csv")
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.csv_file, index=False, encoding='utf-8')
        
        # Create initial prompt
        self.initial_prompt = """
당신은 한국어 문장 분류 전문가입니다.
다음 기준에 따라 문장을 분류하세요:

유형:
- 사실형: 객관적 사실
- 추론형: 분석이나 의견
- 대화형: 구어체나 인사말
- 예측형: 미래에 대한 예측

극성:
- 긍정: 긍정적이거나 중립적
- 부정: 부정적
- 미정: 질문이나 불확실

시제:
- 과거: 과거 시제
- 현재: 현재 시제
- 미래: 미래 시제

확실성:
- 확실: 명확한 사실
- 불확실: 추측이나 가능성

출력 형식: 번호. 유형,극성,시제,확실성
"""
        
        self.prompt_file = os.path.join(self.prompt_dir, "initial_prompt.txt")
        with open(self.prompt_file, 'w', encoding='utf-8') as f:
            f.write(self.initial_prompt)
        
        # Create test config
        self.config = OptimizationConfig(
            gemini_api_key="test_key",
            target_accuracy=0.8,  # Lower target for testing
            max_iterations=3,     # Fewer iterations for testing
            batch_size=10,
            samples_csv_path=self.csv_file,
            prompt_dir=self.prompt_dir,
            analysis_dir=self.analysis_dir
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('services.gemini_client.genai')
    def test_csv_processor_integration(self, mock_genai):
        """Test CSV processor integration with the system"""
        optimizer = GeminiPromptOptimizer(self.config)
        
        # Test data loading
        optimizer._load_and_validate_data()
        
        self.assertEqual(len(optimizer.samples), 5)
        self.assertEqual(optimizer.samples[0].sentence, "오늘은 날씨가 좋다.")
        self.assertEqual(optimizer.samples[0].type, "사실형")
    
    @patch('services.gemini_client.genai')
    def test_prompt_loading_integration(self, mock_genai):
        """Test prompt loading integration"""
        optimizer = GeminiPromptOptimizer(self.config)
        
        # Test prompt loading
        optimizer._load_initial_prompt(self.prompt_file)
        
        self.assertIn("한국어 문장 분류", optimizer.current_prompt)
        self.assertIn("유형:", optimizer.current_prompt)
    
    @patch('services.gemini_client.genai')
    def test_classifier_initialization(self, mock_genai):
        """Test classifier initialization integration"""
        # Mock the genai module
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        optimizer = GeminiPromptOptimizer(self.config)
        optimizer._load_and_validate_data()
        optimizer._load_initial_prompt(self.prompt_file)
        optimizer._initialize_optimization()
        
        self.assertIsNotNone(optimizer.current_classifier)
        self.assertEqual(optimizer.current_classifier.system_prompt, optimizer.current_prompt)
    
    @patch('services.gemini_client.genai')
    def test_result_analysis_integration(self, mock_genai):
        """Test result analysis integration"""
        # Mock API responses
        mock_model = Mock()
        mock_response = Mock()
        mock_response.text = """1. 사실형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실"""
        
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        optimizer = GeminiPromptOptimizer(self.config)
        optimizer._load_and_validate_data()
        optimizer._load_initial_prompt(self.prompt_file)
        optimizer._initialize_optimization()
        
        # Mock the client's generate_content_with_retry method
        optimizer.current_classifier.client.generate_content_with_retry = Mock(return_value=mock_response.text)
        
        # Test classification and analysis
        accuracy, errors = optimizer._test_current_prompt()
        
        self.assertEqual(accuracy, 1.0)  # Perfect accuracy with mocked responses
        self.assertEqual(len(errors), 0)
    
    @patch('services.gemini_client.genai')
    def test_error_analysis_integration(self, mock_genai):
        """Test error analysis integration"""
        # Mock API responses with some errors
        mock_model = Mock()
        mock_classification_response = Mock()
        mock_classification_response.text = """1. 추론형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실"""  # First one is wrong (should be 사실형)
        
        mock_analysis_response = Mock()
        mock_analysis_response.text = """
## 오류 패턴 분석
유형 분류에서 사실형을 추론형으로 잘못 분류하는 패턴이 발견됩니다.

## 문제점 식별
객관적 사실과 주관적 추론을 구분하는 기준이 불명확합니다.

## 개선 제안
- 사실형 분류 기준을 더 명확히 정의
- 객관적 사실의 예시 추가

## 프롬프트 수정 방향
- 사실형 정의 강화
- 예시 문장 추가

## 신뢰도 점수
0.8
"""
        
        mock_model.generate_content.return_value = mock_classification_response
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        optimizer = GeminiPromptOptimizer(self.config)
        optimizer._load_and_validate_data()
        optimizer._load_initial_prompt(self.prompt_file)
        optimizer._initialize_optimization()
        
        # Mock different responses for classification and analysis
        def mock_generate_content_side_effect(model, prompt):
            if "분석" in prompt or "오류" in prompt:
                return mock_analysis_response.text
            else:
                return mock_classification_response.text
        
        optimizer.current_classifier.client.generate_content_with_retry = Mock(
            side_effect=lambda model, prompt: mock_generate_content_side_effect(model, prompt)
        )
        optimizer.pro_analyzer.client.generate_content_with_retry = Mock(
            side_effect=lambda model, prompt: mock_generate_content_side_effect(model, prompt)
        )
        
        # Test with errors
        accuracy, errors = optimizer._test_current_prompt()
        
        self.assertLess(accuracy, 1.0)  # Should have some errors
        self.assertGreater(len(errors), 0)
        
        # Test error analysis
        if errors:
            analysis_report = optimizer._analyze_errors(errors)
            self.assertIsNotNone(analysis_report)
            self.assertGreater(len(analysis_report.improvement_suggestions), 0)
    
    @patch('services.gemini_client.genai')
    def test_prompt_improvement_integration(self, mock_genai):
        """Test prompt improvement integration"""
        # Mock responses
        mock_model = Mock()
        mock_improved_prompt = """
개선된 한국어 문장 분류 시스템입니다.

유형 분류 기준:
- 사실형: 객관적이고 검증 가능한 사실, 통계, 뉴스 보도
- 추론형: 개인의 분석, 의견, 해석, 추측
- 대화형: 직접 인용문, 구어체 표현, 인사말
- 예측형: 미래에 대한 예측, 계획, 전망

극성 분류 기준:
- 긍정: 긍정적 내용 또는 중립적 서술
- 부정: 부정적 내용, 문제점 지적
- 미정: 질문문, 불확실한 추측

시제 분류 기준:
- 과거: 과거 시제 표현
- 현재: 현재 시제, 일반적 사실
- 미래: 미래 시제, 계획

확실성 분류 기준:
- 확실: 명확하고 확정적인 내용
- 불확실: 추측, 가능성 표현

출력 형식: 번호. 유형,극성,시제,확실성
"""
        
        mock_model.generate_content.return_value = Mock(text=mock_improved_prompt)
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        optimizer = GeminiPromptOptimizer(self.config)
        optimizer._load_and_validate_data()
        optimizer._load_initial_prompt(self.prompt_file)
        optimizer._initialize_optimization()
        
        # Mock the improvement process
        optimizer.prompt_optimizer.client.generate_content_with_retry = Mock(
            return_value=mock_improved_prompt
        )
        
        # Create mock analysis report
        from models.data_models import AnalysisReport
        analysis_report = AnalysisReport(
            total_errors=1,
            error_patterns={"type": 1},
            improvement_suggestions=["사실형 기준 명확화"],
            prompt_modifications=["예시 추가"],
            confidence_score=0.8,
            analysis_text="분석 결과"
        )
        
        # Test prompt improvement
        improved_prompt = optimizer._improve_prompt(analysis_report)
        
        self.assertIn("개선된", improved_prompt)
        self.assertIn("사실형:", improved_prompt)
    
    @patch('services.gemini_client.genai')
    def test_iteration_control_integration(self, mock_genai):
        """Test iteration control integration"""
        # Mock perfect responses to test convergence
        mock_model = Mock()
        mock_perfect_response = """1. 사실형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실"""
        
        mock_model.generate_content.return_value = Mock(text=mock_perfect_response)
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        optimizer = GeminiPromptOptimizer(self.config)
        optimizer._load_and_validate_data()
        optimizer._load_initial_prompt(self.prompt_file)
        optimizer._initialize_optimization()
        
        # Mock perfect classification
        optimizer.current_classifier.client.generate_content_with_retry = Mock(
            return_value=mock_perfect_response
        )
        
        # Test iteration control
        optimizer.iteration_controller.start_iteration(1)
        accuracy, errors = optimizer._test_current_prompt()
        optimizer.iteration_controller.update_results(accuracy, len(optimizer.samples) - len(errors), len(errors))
        
        # Should converge immediately with perfect accuracy
        converged = optimizer.iteration_controller.check_convergence()
        self.assertTrue(converged)
    
    def test_file_operations_integration(self):
        """Test file operations integration"""
        # Test that all file operations work together
        optimizer = GeminiPromptOptimizer(self.config)
        
        # Test directory creation
        self.config.create_directories()
        self.assertTrue(os.path.exists(self.config.prompt_dir))
        self.assertTrue(os.path.exists(self.config.analysis_dir))
        
        # Test prompt file operations
        test_prompt = "테스트 프롬프트"
        new_prompt_path = optimizer.prompt_optimizer.save_new_version(
            test_prompt, 
            os.path.join(self.config.prompt_dir, "test_prompt.txt"),
            1
        )
        
        self.assertTrue(os.path.exists(new_prompt_path))
        
        # Test loading the saved prompt
        loaded_prompt = optimizer.prompt_optimizer.load_current_prompt(new_prompt_path)
        self.assertEqual(loaded_prompt, test_prompt)
    
    def test_monitoring_integration(self):
        """Test monitoring system integration"""
        from utils.monitoring import OptimizationMonitor
        from models.data_models import IterationState
        
        monitor = OptimizationMonitor(self.analysis_dir)
        monitor.start_monitoring()
        
        # Create test iteration state
        state = IterationState(
            iteration=1,
            current_accuracy=0.8,
            target_accuracy=0.9,
            best_accuracy=0.8,
            best_prompt_version=1,
            is_converged=False,
            total_samples=5,
            correct_predictions=4,
            error_count=1
        )
        
        # Record metrics
        monitor.record_iteration_metrics(1, state)
        
        # Generate report
        report = monitor.generate_progress_report()
        self.assertIn("Current Accuracy: 0.8000", report)
        
        # Export metrics
        export_path = monitor.export_metrics()
        self.assertTrue(os.path.exists(export_path))

if __name__ == '__main__':
    unittest.main()