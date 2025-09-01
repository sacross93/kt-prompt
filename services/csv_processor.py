"""
CSV data processing service for Gemini Prompt Optimizer
"""
import pandas as pd
from typing import List, Dict, Any, Tuple
from models.data_models import Sample
from models.exceptions import FileProcessingError, ValidationError
from utils.file_utils import read_csv_file
import logging

logger = logging.getLogger("gemini_optimizer.csv_processor")

class CSVProcessor:
    """CSV data processor for samples.csv"""
    
    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._samples: List[Sample] = []
        self._loaded = False
    
    def load_samples(self) -> List[Sample]:
        """Load samples from CSV file"""
        try:
            logger.info(f"Loading samples from {self.csv_path}")
            
            # Read CSV using pandas for better handling
            df = pd.read_csv(self.csv_path, encoding='utf-8')
            
            # Validate required columns
            required_columns = ['id', 'sentence', 'type', 'polarity', 'tense', 'certainty']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValidationError(f"Missing required columns: {missing_columns}")
            
            # Convert to Sample objects
            samples = []
            for _, row in df.iterrows():
                try:
                    sample = Sample(
                        id=int(row['id']),
                        sentence=str(row['sentence']).strip(),
                        type=str(row['type']).strip(),
                        polarity=str(row['polarity']).strip(),
                        tense=str(row['tense']).strip(),
                        certainty=str(row['certainty']).strip()
                    )
                    
                    # Validate sample data
                    self._validate_sample(sample)
                    samples.append(sample)
                    
                except Exception as e:
                    logger.warning(f"Skipping invalid row {row.get('id', 'unknown')}: {e}")
                    continue
            
            if not samples:
                raise ValidationError("No valid samples found in CSV file")
            
            self._samples = samples
            self._loaded = True
            
            logger.info(f"Successfully loaded {len(samples)} samples")
            return samples
            
        except FileNotFoundError:
            raise FileProcessingError(f"CSV file not found: {self.csv_path}")
        except pd.errors.EmptyDataError:
            raise FileProcessingError(f"CSV file is empty: {self.csv_path}")
        except Exception as e:
            raise FileProcessingError(f"Failed to load CSV file {self.csv_path}: {e}")
    
    def _validate_sample(self, sample: Sample) -> None:
        """Validate individual sample data"""
        # Check if sentence is not empty
        if not sample.sentence or sample.sentence.isspace():
            raise ValidationError(f"Empty sentence for sample {sample.id}")
        
        # Validate classification values
        valid_types = ["사실형", "추론형", "대화형", "예측형"]
        valid_polarities = ["긍정", "부정", "미정"]
        valid_tenses = ["과거", "현재", "미래"]
        valid_certainties = ["확실", "불확실"]
        
        if sample.type not in valid_types:
            raise ValidationError(f"Invalid type '{sample.type}' for sample {sample.id}")
        
        if sample.polarity not in valid_polarities:
            raise ValidationError(f"Invalid polarity '{sample.polarity}' for sample {sample.id}")
        
        if sample.tense not in valid_tenses:
            raise ValidationError(f"Invalid tense '{sample.tense}' for sample {sample.id}")
        
        if sample.certainty not in valid_certainties:
            raise ValidationError(f"Invalid certainty '{sample.certainty}' for sample {sample.id}")
    
    def extract_questions(self, samples: List[Sample] = None) -> List[str]:
        """Extract questions (sentences) from samples"""
        if samples is None:
            if not self._loaded:
                samples = self.load_samples()
            else:
                samples = self._samples
        
        questions = []
        for i, sample in enumerate(samples, 1):
            # Format: "1. sentence"
            questions.append(f"{i}. {sample.sentence}")
        
        logger.info(f"Extracted {len(questions)} questions")
        return questions
    
    def get_correct_answers(self, samples: List[Sample] = None) -> List[str]:
        """Get correct answers from samples"""
        if samples is None:
            if not self._loaded:
                samples = self.load_samples()
            else:
                samples = self._samples
        
        answers = []
        for i, sample in enumerate(samples, 1):
            # Format: "1. type,polarity,tense,certainty"
            answer = f"{i}. {sample.get_expected_output()}"
            answers.append(answer)
        
        logger.info(f"Extracted {len(answers)} correct answers")
        return answers
    
    def get_samples_batch(self, batch_size: int, start_idx: int = 0) -> List[Sample]:
        """Get batch of samples"""
        if not self._loaded:
            self.load_samples()
        
        end_idx = min(start_idx + batch_size, len(self._samples))
        batch = self._samples[start_idx:end_idx]
        
        logger.debug(f"Retrieved batch of {len(batch)} samples (indices {start_idx}-{end_idx-1})")
        return batch
    
    def get_total_samples_count(self) -> int:
        """Get total number of samples"""
        if not self._loaded:
            self.load_samples()
        return len(self._samples)
    
    def get_sample_by_id(self, sample_id: int) -> Sample:
        """Get sample by ID"""
        if not self._loaded:
            self.load_samples()
        
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        
        raise ValidationError(f"Sample with ID {sample_id} not found")
    
    def get_samples_by_ids(self, sample_ids: List[int]) -> List[Sample]:
        """Get multiple samples by IDs"""
        if not self._loaded:
            self.load_samples()
        
        samples = []
        for sample_id in sample_ids:
            try:
                sample = self.get_sample_by_id(sample_id)
                samples.append(sample)
            except ValidationError:
                logger.warning(f"Sample ID {sample_id} not found, skipping")
                continue
        
        return samples
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics"""
        if not self._loaded:
            self.load_samples()
        
        stats = {
            "total_samples": len(self._samples),
            "type_distribution": {},
            "polarity_distribution": {},
            "tense_distribution": {},
            "certainty_distribution": {},
            "sentence_length_stats": {}
        }
        
        # Count distributions
        for sample in self._samples:
            # Type distribution
            stats["type_distribution"][sample.type] = stats["type_distribution"].get(sample.type, 0) + 1
            
            # Polarity distribution
            stats["polarity_distribution"][sample.polarity] = stats["polarity_distribution"].get(sample.polarity, 0) + 1
            
            # Tense distribution
            stats["tense_distribution"][sample.tense] = stats["tense_distribution"].get(sample.tense, 0) + 1
            
            # Certainty distribution
            stats["certainty_distribution"][sample.certainty] = stats["certainty_distribution"].get(sample.certainty, 0) + 1
        
        # Sentence length statistics
        lengths = [len(sample.sentence) for sample in self._samples]
        if lengths:
            stats["sentence_length_stats"] = {
                "min": min(lengths),
                "max": max(lengths),
                "avg": sum(lengths) / len(lengths),
                "median": sorted(lengths)[len(lengths) // 2]
            }
        
        return stats
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """Validate entire dataset and return validation results"""
        if not self._loaded:
            self.load_samples()
        
        errors = []
        
        # Check for duplicate IDs
        ids = [sample.id for sample in self._samples]
        if len(ids) != len(set(ids)):
            errors.append("Duplicate sample IDs found")
        
        # Check for empty sentences
        empty_sentences = [sample.id for sample in self._samples if not sample.sentence.strip()]
        if empty_sentences:
            errors.append(f"Empty sentences found in samples: {empty_sentences}")
        
        # Check classification consistency
        for sample in self._samples:
            try:
                self._validate_sample(sample)
            except ValidationError as e:
                errors.append(str(e))
        
        is_valid = len(errors) == 0
        
        if is_valid:
            logger.info("Dataset validation passed")
        else:
            logger.warning(f"Dataset validation failed with {len(errors)} errors")
            for error in errors:
                logger.warning(f"  - {error}")
        
        return is_valid, errors