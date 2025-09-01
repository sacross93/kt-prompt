"""
End-to-end tests for Gemini Prompt Optimizer
"""
import unittest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig

class TestE2EOptimization(unittest.TestCase):
    """End-to-end tests for complete optimization workflow"""
    
    def setUp(self):
        """Set up test fixtures for E2E testing"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.prompt_dir = os.path.join(self.temp_dir, "prompts")
        self.analysis_dir = os.path.join(self.temp_dir, "analysis")
        
        os.makedirs(self.prompt_dir, exist_ok=True)
        os.makedirs(self.analysis_dir, exist_ok=True)
        
        # Create larger test dataset for E2E testing
        self.test_data = []
        sentences_data = [
            ("오늘은 날씨가 좋다.", "사실형", "긍정", "현재", "확실"),
            ("내일 비가 올 것 같다.", "예측형", "미정", "미래", "불확실"),
            ("어제 영화를 봤다.", "사실형", "긍정", "과거", "확실"),
            ("이 영화는 재미없을 것이다.", "예측형", "부정", "미래", "불확실"),
            ("안녕하세요, 반갑습니다.", "대화형", "긍정", "현재", "확실"),
            ("그는 똑똑한 사람인 것 같다.", "추론형", "긍정", "현재", "불확실"),
            ("회의는 어제 끝났다.", "사실형", "긍정", "과거", "확실"),
            ("내년에 새로운 프로젝트를 시작할 예정이다.", "예측형", "긍정", "미래", "확실"),
            ("이 문제는 해결하기 어렵다.", "추론형", "부정", "현재", "확실"),
            ("\"좋은 아침입니다\"라고 인사했다.", "대화형", "긍정", "과거", "확실")
        ]
        
        for i, (sentence, type_, polarity, tense, certainty) in enumerate(sentences_data, 1):
            self.test_data.append({
                "id": i,
                "sentence": sentence,
                "type": type_,
                "polarity": polarity,
                "tense": tense,
                "certainty": certainty
            })
        
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
        
        # Create test config with realistic settings
        self.config = OptimizationConfig(
            gemini_api_key="test_key",
            target_accuracy=0.9,
            max_iterations=3,
            batch_size=5,
            samples_csv_path=self.csv_file,
            prompt_dir=self.prompt_dir,
            analysis_dir=self.analysis_dir,
            convergence_threshold=0.01,
            patience=2
        )
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('services.gemini_client.genai')
    def test_complete_optimization_workflow_with_improvement(self, mock_genai):
        """Test complete optimization workflow with gradual improvement"""
        
        # Mock responses that show gradual improvement
        iteration_responses = [
            # Iteration 1: 70% accuracy (3 errors)
            """1. 추론형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실
6. 추론형,긍정,현재,불확실
7. 사실형,긍정,과거,확실
8. 예측형,긍정,미래,확실
9. 추론형,부정,현재,확실
10. 대화형,긍정,과거,확실""",
            
            # Iteration 2: 80% accuracy (2 errors)
            """1. 사실형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실
6. 추론형,긍정,현재,불확실
7. 사실형,긍정,과거,확실
8. 예측형,긍정,미래,확실
9. 추론형,부정,현재,확실
10. 대화형,긍정,과거,확실""",
            
            # Iteration 3: 90% accuracy (1 error) - reaches target
            """1. 사실형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실
6. 추론형,긍정,현재,불확실
7. 사실형,긍정,과거,확실
8. 예측형,긍정,미래,확실
9. 추론형,부정,현재,확실
10. 대화형,긍정,과거,확실"""
        ]
        
        # Mock analysis responses
        analysis_responses = [
            """
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
""",
            """
## 오류 패턴 분석
시제 분류에서 일부 오류가 발견됩니다.

## 문제점 식별
미래 시제와 현재 시제 구분이 불명확합니다.

## 개선 제안
- 시제 분류 기준 세분화
- 미래 표현 패턴 명시

## 프롬프트 수정 방향
- 시제 구분 기준 강화

## 신뢰도 점수
0.9
"""
        ]
        
        # Mock improved prompts
        improved_prompts = [
            """
개선된 한국어 문장 분류 시스템입니다.

유형 분류 기준:
- 사실형: 객관적이고 검증 가능한 사실, 통계, 뉴스 보도
- 추론형: 개인의 분석, 의견, 해석, 추측
- 대화형: 직접 인용문, 구어체 표현, 인사말
- 예측형: 미래에 대한 예측, 계획, 전망

극성, 시제, 확실성 기준은 동일합니다.
출력 형식: 번호. 유형,극성,시제,확실성
""",
            """
더욱 개선된 한국어 문장 분류 시스템입니다.

유형 분류 기준:
- 사실형: 객관적이고 검증 가능한 사실, 통계, 뉴스 보도
- 추론형: 개인의 분석, 의견, 해석, 추측
- 대화형: 직접 인용문, 구어체 표현, 인사말
- 예측형: 미래에 대한 예측, 계획, 전망

시제 분류 기준 (강화):
- 과거: "~했다", "~였다" 등 명확한 과거 표현
- 현재: "~이다", "~한다" 등 현재 상태나 일반적 사실
- 미래: "~할 것이다", "~예정" 등 미래 계획이나 예측

출력 형식: 번호. 유형,극성,시제,확실성
"""
        ]
        
        # Set up mocks
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        # Create response iterator
        response_iter = iter(iteration_responses + analysis_responses + improved_prompts)
        
        def mock_generate_content_side_effect(prompt):
            try:
                return Mock(text=next(response_iter))
            except StopIteration:
                return Mock(text="Default response")
        
        mock_model.generate_content.side_effect = mock_generate_content_side_effect
        
        # Create optimizer and run
        optimizer = GeminiPromptOptimizer(self.config)
        
        # Mock the client methods
        def mock_client_generate(model, prompt):
            return mock_generate_content_side_effect(prompt).text
        
        # Patch the client methods after optimizer creation
        with patch.object(optimizer, 'current_classifier', None):
            with patch('services.gemini_flash_classifier.GeminiClient') as mock_flash_client:
                with patch('services.gemini_pro_analyzer.GeminiClient') as mock_pro_client:
                    with patch('services.prompt_optimizer.GeminiClient') as mock_opt_client:
                        
                        # Configure mock clients
                        for mock_client_class in [mock_flash_client, mock_pro_client, mock_opt_client]:
                            mock_client_instance = Mock()
                            mock_client_instance.generate_content_with_retry = Mock(
                                side_effect=mock_client_generate
                            )
                            mock_client_instance.get_flash_model.return_value = mock_model
                            mock_client_instance.get_pro_model.return_value = mock_model
                            mock_client_class.return_value = mock_client_instance
                        
                        # Run optimization
                        result = optimizer.run_optimization(self.prompt_file)
                        
                        # Verify results
                        self.assertIsNotNone(result)
                        self.assertGreaterEqual(result.final_accuracy, 0.7)  # Should improve over iterations
                        self.assertGreater(result.total_iterations, 0)
                        self.assertLess(result.total_iterations, self.config.max_iterations + 1)
    
    @patch('services.gemini_client.genai')
    def test_early_convergence_scenario(self, mock_genai):
        """Test scenario where optimization converges early"""
        
        # Mock perfect response from the start
        perfect_response = """1. 사실형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실
6. 추론형,긍정,현재,불확실
7. 사실형,긍정,과거,확실
8. 예측형,긍정,미래,확실
9. 추론형,부정,현재,확실
10. 대화형,긍정,과거,확실"""
        
        # Set up mocks
        mock_model = Mock()
        mock_model.generate_content.return_value = Mock(text=perfect_response)
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        # Create optimizer with lower target for early convergence
        config = self.config
        config.target_accuracy = 0.8  # Lower target
        
        optimizer = GeminiPromptOptimizer(config)
        
        # Mock client methods
        with patch('services.gemini_flash_classifier.GeminiClient') as mock_client:
            mock_client_instance = Mock()
            mock_client_instance.generate_content_with_retry = Mock(return_value=perfect_response)
            mock_client_instance.get_flash_model.return_value = mock_model
            mock_client.return_value = mock_client_instance
            
            # Run optimization
            result = optimizer.run_optimization(self.prompt_file)
            
            # Should converge in first iteration due to perfect accuracy
            self.assertEqual(result.total_iterations, 1)
            self.assertTrue(result.convergence_achieved)
            self.assertEqual(result.final_accuracy, 1.0)
    
    @patch('services.gemini_client.genai')
    def test_max_iterations_scenario(self, mock_genai):
        """Test scenario where optimization reaches max iterations"""
        
        # Mock responses that never reach target
        mediocre_response = """1. 추론형,긍정,현재,확실
2. 예측형,미정,미래,불확실
3. 사실형,긍정,과거,확실
4. 예측형,부정,미래,불확실
5. 대화형,긍정,현재,확실
6. 추론형,긍정,현재,불확실
7. 사실형,긍정,과거,확실
8. 예측형,긍정,미래,확실
9. 추론형,부정,현재,확실
10. 대화형,긍정,과거,확실"""  # 70% accuracy, never improves
        
        mock_analysis = """
## 오류 패턴 분석
지속적인 분류 오류가 발견됩니다.

## 개선 제안
- 기준 재정의 필요

## 프롬프트 수정 방향
- 전면 수정 필요

## 신뢰도 점수
0.5
"""
        
        mock_improved_prompt = "개선 시도했지만 여전히 부족한 프롬프트"
        
        # Set up mocks
        mock_model = Mock()
        mock_genai.GenerativeModel.return_value = mock_model
        mock_genai.configure = Mock()
        mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
        
        # Always return the same mediocre response
        responses = [mediocre_response, mock_analysis, mock_improved_prompt] * 10
        response_iter = iter(responses)
        
        mock_model.generate_content.side_effect = lambda prompt: Mock(text=next(response_iter))
        
        # Create optimizer with high target that won't be reached
        config = self.config
        config.target_accuracy = 0.99  # Very high target
        config.max_iterations = 2      # Low max iterations
        
        optimizer = GeminiPromptOptimizer(config)
        
        # Mock client methods
        def mock_client_generate(model, prompt):
            return next(response_iter)
        
        with patch('services.gemini_flash_classifier.GeminiClient') as mock_flash_client:
            with patch('services.gemini_pro_analyzer.GeminiClient') as mock_pro_client:
                with patch('services.prompt_optimizer.GeminiClient') as mock_opt_client:
                    
                    # Configure all mock clients
                    for mock_client_class in [mock_flash_client, mock_pro_client, mock_opt_client]:
                        mock_client_instance = Mock()
                        mock_client_instance.generate_content_with_retry = Mock(
                            side_effect=mock_client_generate
                        )
                        mock_client_instance.get_flash_model.return_value = mock_model
                        mock_client_instance.get_pro_model.return_value = mock_model
                        mock_client_class.return_value = mock_client_instance
                    
                    # Run optimization
                    result = optimizer.run_optimization(self.prompt_file)
                    
                    # Should reach max iterations
                    self.assertEqual(result.total_iterations, config.max_iterations)
                    self.assertFalse(result.convergence_achieved)
                    self.assertLess(result.final_accuracy, config.target_accuracy)
    
    def test_file_persistence_e2e(self):
        """Test that all files are properly created and persisted"""
        
        # Create a minimal mock setup
        with patch('services.gemini_client.genai') as mock_genai:
            mock_model = Mock()
            mock_model.generate_content.return_value = Mock(text="1. 사실형,긍정,현재,확실\n2. 예측형,미정,미래,불확실")
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.configure = Mock()
            mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
            
            # Lower target for quick convergence
            config = self.config
            config.target_accuracy = 0.5
            config.max_iterations = 1
            
            optimizer = GeminiPromptOptimizer(config)
            
            with patch('services.gemini_flash_classifier.GeminiClient') as mock_client:
                mock_client_instance = Mock()
                mock_client_instance.generate_content_with_retry = Mock(
                    return_value="1. 사실형,긍정,현재,확실\n2. 예측형,미정,미래,불확실"
                )
                mock_client_instance.get_flash_model.return_value = mock_model
                mock_client.return_value = mock_client_instance
                
                # Run optimization
                result = optimizer.run_optimization(self.prompt_file)
                
                # Check that files were created
                self.assertTrue(os.path.exists(result.final_prompt_path))
                
                # Check log files
                log_files = [f for f in os.listdir(self.analysis_dir) if f.endswith('.log')]
                self.assertGreater(len(log_files), 0)
                
                # Check that directories exist
                self.assertTrue(os.path.exists(self.prompt_dir))
                self.assertTrue(os.path.exists(self.analysis_dir))
    
    def test_error_handling_e2e(self):
        """Test error handling in E2E scenario"""
        
        # Test with invalid CSV file
        invalid_csv = os.path.join(self.temp_dir, "invalid.csv")
        with open(invalid_csv, 'w') as f:
            f.write("invalid,csv,format\n")
        
        config = self.config
        config.samples_csv_path = invalid_csv
        
        optimizer = GeminiPromptOptimizer(config)
        
        # Should raise an error due to invalid CSV
        with self.assertRaises(Exception):
            optimizer.run_optimization(self.prompt_file)
        
        # Test with missing prompt file
        config.samples_csv_path = self.csv_file  # Reset to valid CSV
        
        with self.assertRaises(FileNotFoundError):
            optimizer.run_optimization("nonexistent_prompt.txt")
    
    def test_monitoring_e2e(self):
        """Test monitoring integration in E2E scenario"""
        
        with patch('services.gemini_client.genai') as mock_genai:
            mock_model = Mock()
            mock_model.generate_content.return_value = Mock(text="Perfect response")
            mock_genai.GenerativeModel.return_value = mock_model
            mock_genai.configure = Mock()
            mock_genai.list_models.return_value = [Mock(name="gemini-1.5-flash")]
            
            config = self.config
            config.target_accuracy = 0.5  # Low target for quick completion
            
            optimizer = GeminiPromptOptimizer(config)
            
            # Add monitoring
            from utils.monitoring import OptimizationMonitor
            monitor = OptimizationMonitor(self.analysis_dir)
            
            with patch('services.gemini_flash_classifier.GeminiClient') as mock_client:
                mock_client_instance = Mock()
                mock_client_instance.generate_content_with_retry = Mock(return_value="Perfect response")
                mock_client_instance.get_flash_model.return_value = mock_model
                mock_client.return_value = mock_client_instance
                
                # Run with monitoring
                monitor.start_monitoring()
                result = optimizer.run_optimization(self.prompt_file)
                
                # Export monitoring results
                export_path = monitor.export_metrics()
                self.assertTrue(os.path.exists(export_path))

if __name__ == '__main__':
    unittest.main()