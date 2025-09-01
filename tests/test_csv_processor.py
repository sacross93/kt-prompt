"""
Unit tests for CSV processor
"""
import unittest
import tempfile
import os
import pandas as pd
from services.csv_processor import CSVProcessor
from models.data_models import Sample
from models.exceptions import FileProcessingError, ValidationError

class TestCSVProcessor(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data = [
            {"id": 1, "sentence": "오늘은 날씨가 좋다.", "type": "사실형", "polarity": "긍정", "tense": "현재", "certainty": "확실"},
            {"id": 2, "sentence": "내일 비가 올 것 같다.", "type": "예측형", "polarity": "미정", "tense": "미래", "certainty": "불확실"},
            {"id": 3, "sentence": "어제 영화를 봤다.", "type": "사실형", "polarity": "긍정", "tense": "과거", "certainty": "확실"}
        ]
        
        # Create temporary CSV file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        df = pd.DataFrame(self.test_data)
        df.to_csv(self.temp_file.name, index=False, encoding='utf-8')
        self.temp_file.close()
        
        self.processor = CSVProcessor(self.temp_file.name)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_file.name):
            os.unlink(self.temp_file.name)
    
    def test_load_samples_success(self):
        """Test successful sample loading"""
        samples = self.processor.load_samples()
        
        self.assertEqual(len(samples), 3)
        self.assertIsInstance(samples[0], Sample)
        self.assertEqual(samples[0].id, 1)
        self.assertEqual(samples[0].sentence, "오늘은 날씨가 좋다.")
        self.assertEqual(samples[0].type, "사실형")
    
    def test_load_samples_file_not_found(self):
        """Test loading from non-existent file"""
        processor = CSVProcessor("nonexistent.csv")
        
        with self.assertRaises(FileProcessingError):
            processor.load_samples()
    
    def test_extract_questions(self):
        """Test question extraction"""
        samples = self.processor.load_samples()
        questions = self.processor.extract_questions(samples)
        
        self.assertEqual(len(questions), 3)
        self.assertEqual(questions[0], "1. 오늘은 날씨가 좋다.")
        self.assertEqual(questions[1], "2. 내일 비가 올 것 같다.")
    
    def test_get_correct_answers(self):
        """Test correct answer extraction"""
        samples = self.processor.load_samples()
        answers = self.processor.get_correct_answers(samples)
        
        self.assertEqual(len(answers), 3)
        self.assertEqual(answers[0], "1. 사실형,긍정,현재,확실")
        self.assertEqual(answers[1], "2. 예측형,미정,미래,불확실")
    
    def test_get_samples_batch(self):
        """Test batch sample retrieval"""
        samples = self.processor.load_samples()
        batch = self.processor.get_samples_batch(2, 0)
        
        self.assertEqual(len(batch), 2)
        self.assertEqual(batch[0].id, 1)
        self.assertEqual(batch[1].id, 2)
    
    def test_get_sample_by_id(self):
        """Test sample retrieval by ID"""
        self.processor.load_samples()
        sample = self.processor.get_sample_by_id(2)
        
        self.assertEqual(sample.id, 2)
        self.assertEqual(sample.sentence, "내일 비가 올 것 같다.")
    
    def test_get_sample_by_id_not_found(self):
        """Test sample retrieval with invalid ID"""
        self.processor.load_samples()
        
        with self.assertRaises(ValidationError):
            self.processor.get_sample_by_id(999)
    
    def test_get_statistics(self):
        """Test dataset statistics"""
        self.processor.load_samples()
        stats = self.processor.get_statistics()
        
        self.assertEqual(stats["total_samples"], 3)
        self.assertEqual(stats["type_distribution"]["사실형"], 2)
        self.assertEqual(stats["type_distribution"]["예측형"], 1)
        self.assertIn("sentence_length_stats", stats)
    
    def test_validate_dataset_success(self):
        """Test successful dataset validation"""
        self.processor.load_samples()
        is_valid, errors = self.processor.validate_dataset()
        
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validate_sample_invalid_type(self):
        """Test sample validation with invalid type"""
        invalid_sample = Sample(
            id=1, sentence="Test", type="invalid_type", 
            polarity="긍정", tense="현재", certainty="확실"
        )
        
        with self.assertRaises(ValidationError):
            self.processor._validate_sample(invalid_sample)
    
    def test_validate_sample_empty_sentence(self):
        """Test sample validation with empty sentence"""
        invalid_sample = Sample(
            id=1, sentence="", type="사실형", 
            polarity="긍정", tense="현재", certainty="확실"
        )
        
        with self.assertRaises(ValidationError):
            self.processor._validate_sample(invalid_sample)

if __name__ == '__main__':
    unittest.main()