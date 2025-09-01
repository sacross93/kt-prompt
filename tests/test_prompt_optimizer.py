"""
Unit tests for prompt optimizer
"""
import unittest
import tempfile
import os
from unittest.mock import Mock, patch
from services.prompt_optimizer import PromptOptimizer
from models.data_models import AnalysisReport
from config import OptimizationConfig

class TestPromptOptimizer(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create mock config
        self.config = Mock(spec=OptimizationConfig)
        self.config.prompt_dir = tempfile.mkdtemp()
        self.config.analysis_dir = tempfile.mkdtemp()
        self.config.gemini_api_key = "test_key"
        self.config.pro_model = "gemini-1.5-pro"
        
        # Create test prompt file
        self.test_prompt = """
당신은 한국어 문장 분류 전문가입니다.
다음 기준에 따라 문장을 분류하세요:
- 유형: 사실형, 추론형, 대화형, 예측형
- 극성: 긍정, 부정, 미정
- 시제: 과거, 현재, 미래
- 확실성: 확실, 불확실
"""
        
        self.test_prompt_file = os.path.join(self.config.prompt_dir, "test_prompt.txt")
        with open(self.test_prompt_file, 'w', encoding='utf-8') as f:
            f.write(self.test_prompt)
        
        # Mock the Gemini client
        with patch('services.prompt_optimizer.GeminiClient'):
            with patch('services.prompt_optimizer.GeminiProAnalyzer'):
                self.optimizer = PromptOptimizer(self.config)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.config.prompt_dir, ignore_errors=True)
        shutil.rmtree(self.config.analysis_dir, ignore_errors=True)
    
    def test_load_current_prompt(self):
        """Test loading current prompt"""
        prompt = self.optimizer.load_current_prompt(self.test_prompt_file)
        
        self.assertIn("한국어 문장 분류", prompt)
        self.assertIn("유형:", prompt)
    
    def test_load_current_prompt_file_not_found(self):
        """Test loading non-existent prompt file"""
        with self.assertRaises(Exception):
            self.optimizer.load_current_prompt("nonexistent.txt")
    
    @patch('services.prompt_optimizer.GeminiClient')
    def test_apply_improvements(self, mock_client_class):
        """Test applying improvements to prompt"""
        # Mock the client and model
        mock_client = Mock()
        mock_model = Mock()
        mock_client.get_pro_model.return_value = mock_model
        mock_client.generate_content_with_retry.return_value = "개선된 프롬프트 내용"
        mock_client_class.return_value = mock_client
        
        # Create optimizer with mocked client
        optimizer = PromptOptimizer(self.config)
        optimizer.client = mock_client
        optimizer.model = mock_model
        
        # Create test analysis report
        analysis_report = AnalysisReport(
            total_errors=5,
            error_patterns={"type": 3, "polarity": 2},
            improvement_suggestions=["유형 분류 기준을 명확히 하세요"],
            prompt_modifications=["예시를 추가하세요"],
            confidence_score=0.8,
            analysis_text="분석 결과"
        )
        
        improved_prompt = optimizer.apply_improvements(self.test_prompt, analysis_report)
        
        self.assertEqual(improved_prompt, "개선된 프롬프트 내용")
        mock_client.generate_content_with_retry.assert_called_once()
    
    def test_save_new_version(self):
        """Test saving new prompt version"""
        test_prompt = "새로운 프롬프트 내용"
        
        new_path = self.optimizer.save_new_version(test_prompt, self.test_prompt_file)
        
        self.assertTrue(os.path.exists(new_path))
        self.assertIn("_v1", new_path)
        
        with open(new_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertEqual(content, test_prompt)
    
    def test_log_changes(self):
        """Test logging changes"""
        changes = ["변경사항 1", "변경사항 2"]
        
        self.optimizer.log_changes(changes, 1)
        
        log_file = os.path.join(self.config.analysis_dir, "prompt_changes.log")
        self.assertTrue(os.path.exists(log_file))
        
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        self.assertIn("변경사항 1", content)
        self.assertIn("변경사항 2", content)
    
    def test_backup_current_prompt(self):
        """Test backing up current prompt"""
        backup_path = self.optimizer.backup_current_prompt(self.test_prompt_file)
        
        self.assertTrue(os.path.exists(backup_path))
        self.assertIn("backup", backup_path)
    
    def test_clean_prompt(self):
        """Test prompt cleaning"""
        dirty_prompt = """
        개선된 시스템 프롬프트:
        
        
        실제 프롬프트 내용입니다.
        
        
        """
        
        clean_prompt = self.optimizer._clean_prompt(dirty_prompt)
        
        self.assertEqual(clean_prompt, "실제 프롬프트 내용입니다.")
    
    def test_validate_prompt_success(self):
        """Test successful prompt validation"""
        valid_prompt = """
        한국어 문장을 분류하는 시스템입니다.
        유형: 사실형, 추론형, 대화형, 예측형
        극성: 긍정, 부정, 미정
        시제: 과거, 현재, 미래
        확실성: 확실, 불확실
        """
        
        is_valid = self.optimizer._validate_prompt(valid_prompt)
        self.assertTrue(is_valid)
    
    def test_validate_prompt_too_short(self):
        """Test validation of too short prompt"""
        short_prompt = "짧은 프롬프트"
        
        is_valid = self.optimizer._validate_prompt(short_prompt)
        self.assertFalse(is_valid)
    
    def test_validate_prompt_missing_keywords(self):
        """Test validation of prompt missing essential keywords"""
        incomplete_prompt = "이것은 긴 프롬프트이지만 필수 키워드가 없습니다. " * 10
        
        is_valid = self.optimizer._validate_prompt(incomplete_prompt)
        self.assertFalse(is_valid)
    
    def test_calculate_korean_ratio(self):
        """Test Korean character ratio calculation"""
        text = "한국어 텍스트입니다. English text."
        ratio = self.optimizer.calculate_korean_ratio(text)
        
        # Should be around 0.6-0.7 (Korean characters / total characters)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 0.8)
    
    def test_optimize_korean_ratio(self):
        """Test Korean ratio optimization"""
        text = "This is a system prompt for classification."
        optimized = self.optimizer.optimize_korean_ratio(text)
        
        self.assertIn("시스템", optimized)
        self.assertIn("분류", optimized)
    
    def test_get_prompt_statistics(self):
        """Test prompt statistics calculation"""
        stats = self.optimizer.get_prompt_statistics(self.test_prompt)
        
        self.assertIn("total_length", stats)
        self.assertIn("korean_ratio", stats)
        self.assertIn("line_count", stats)
        self.assertIn("word_count", stats)
        self.assertIn("has_essential_keywords", stats)
    
    def test_fix_type_classification(self):
        """Test type classification fix"""
        prompt = "유형: 기본 설명"
        fixed_prompt = self.optimizer._fix_type_classification(prompt)
        
        self.assertIn("사실형:", fixed_prompt)
        self.assertIn("추론형:", fixed_prompt)
        self.assertIn("대화형:", fixed_prompt)
        self.assertIn("예측형:", fixed_prompt)
    
    def test_fix_polarity_classification(self):
        """Test polarity classification fix"""
        prompt = "극성: 기본 설명"
        fixed_prompt = self.optimizer._fix_polarity_classification(prompt)
        
        self.assertIn("긍정:", fixed_prompt)
        self.assertIn("부정:", fixed_prompt)
        self.assertIn("미정:", fixed_prompt)

if __name__ == '__main__':
    unittest.main()