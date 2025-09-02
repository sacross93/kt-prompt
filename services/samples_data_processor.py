"""
샘플 데이터 처리기

samples.csv 전용 데이터 분석 및 가공/증강/변형
"""

import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
import random
from collections import Counter

logger = logging.getLogger(__name__)

@dataclass
class SamplesAnalysis:
    """샘플 데이터 분석 결과"""
    total_samples: int
    attribute_distributions: Dict[str, Dict[str, int]]  # 속성별 분포
    pattern_examples: Dict[str, List[str]]              # 패턴별 예시들
    boundary_cases: List['Sample']                      # 경계 사례들
    representative_samples: Dict[str, List['Sample']]   # 대표 샘플들
    augmentation_candidates: List['Sample']             # 증강 후보들

@dataclass
class Sample:
    """샘플 데이터"""
    sentence: str
    type: str
    polarity: str
    tense: str
    certainty: str
    
    def to_dict(self) -> Dict[str, str]:
        return {
            'sentence': self.sentence,
            'type': self.type,
            'polarity': self.polarity,
            'tense': self.tense,
            'certainty': self.certainty
        }

class SamplesDataProcessor:
    """샘플 데이터 처리기"""
    
    def __init__(self, csv_path: str = "data/samples.csv"):
        self.csv_path = csv_path
        self.samples = []
        self.analysis_result = None
        
    def analyze_samples_csv(self, csv_path: str = None) -> SamplesAnalysis:
        """samples.csv 분석"""
        if csv_path:
            self.csv_path = csv_path
            
        try:
            df = pd.read_csv(self.csv_path)
            logger.info(f"샘플 데이터 로드: {len(df)}개")
            
            # Sample 객체로 변환
            self.samples = []
            for _, row in df.iterrows():
                # output 컬럼을 파싱하여 속성 추출
                output_parts = row['output'].strip('"').split(',')
                if len(output_parts) >= 4:
                    sample = Sample(
                        sentence=row['user_prompt'],
                        type=output_parts[0],
                        polarity=output_parts[1],
                        tense=output_parts[2],
                        certainty=output_parts[3]
                    )
                    self.samples.append(sample)
            
            # 분석 수행
            analysis = SamplesAnalysis(
                total_samples=len(self.samples),
                attribute_distributions=self._analyze_distributions(),
                pattern_examples=self._extract_pattern_examples(),
                boundary_cases=self._identify_boundary_cases(),
                representative_samples=self._select_representative_samples(),
                augmentation_candidates=self._identify_augmentation_candidates()
            )
            
            self.analysis_result = analysis
            logger.info("샘플 데이터 분석 완료")
            return analysis
            
        except Exception as e:
            logger.error(f"샘플 데이터 분석 실패: {e}")
            raise
    
    def _analyze_distributions(self) -> Dict[str, Dict[str, int]]:
        """속성별 분포 분석"""
        distributions = {}
        
        attributes = ['type', 'polarity', 'tense', 'certainty']
        
        for attr in attributes:
            values = [getattr(sample, attr) for sample in self.samples]
            distributions[attr] = dict(Counter(values))
        
        logger.info(f"속성별 분포: {distributions}")
        return distributions
    
    def _extract_pattern_examples(self) -> Dict[str, List[str]]:
        """패턴별 예시 추출"""
        patterns = {}
        
        # 각 속성 조합별 예시 수집
        for sample in self.samples:
            # 속성 조합을 키로 사용
            key = f"{sample.type}_{sample.polarity}_{sample.tense}_{sample.certainty}"
            
            if key not in patterns:
                patterns[key] = []
            
            patterns[key].append(sample.sentence)
        
        # 각 패턴별로 최대 3개 예시만 유지
        for key in patterns:
            patterns[key] = patterns[key][:3]
        
        logger.info(f"패턴 종류: {len(patterns)}개")
        return patterns    

    def _identify_boundary_cases(self) -> List[Sample]:
        """경계 사례 식별"""
        boundary_cases = []
        
        # 애매한 표현이 포함된 문장들 찾기
        ambiguous_keywords = [
            '아마도', '혹시', '것 같다', '듯하다', '인 듯',
            '좀', '약간', '조금', '어느 정도', '그럭저럭',
            '나름', '어떻게 보면', '한편으로는'
        ]
        
        for sample in self.samples:
            sentence_lower = sample.sentence.lower()
            
            # 애매한 키워드가 포함된 경우
            if any(keyword in sentence_lower for keyword in ambiguous_keywords):
                boundary_cases.append(sample)
            
            # 문장이 너무 짧거나 긴 경우
            elif len(sample.sentence) < 10 or len(sample.sentence) > 100:
                boundary_cases.append(sample)
        
        logger.info(f"경계 사례: {len(boundary_cases)}개")
        return boundary_cases[:10]  # 최대 10개
    
    def _select_representative_samples(self) -> Dict[str, List[Sample]]:
        """대표 샘플 선택"""
        representative = {}
        
        # 각 속성별로 대표 샘플 선택
        attributes = ['type', 'polarity', 'tense', 'certainty']
        
        for attr in attributes:
            attr_samples = {}
            
            # 속성값별로 그룹화
            for sample in self.samples:
                attr_value = getattr(sample, attr)
                if attr_value not in attr_samples:
                    attr_samples[attr_value] = []
                attr_samples[attr_value].append(sample)
            
            # 각 속성값별로 대표 샘플 선택
            representative[attr] = []
            for attr_value, samples in attr_samples.items():
                # 문장 길이가 적당한 것 선택 (30-60자)
                suitable_samples = [s for s in samples if 30 <= len(s.sentence) <= 60]
                if suitable_samples:
                    representative[attr].append(random.choice(suitable_samples))
                elif samples:
                    representative[attr].append(random.choice(samples))
        
        return representative
    
    def _identify_augmentation_candidates(self) -> List[Sample]:
        """증강 후보 식별"""
        candidates = []
        
        # 분포가 적은 속성 조합 찾기
        combination_counts = Counter()
        
        for sample in self.samples:
            combination = (sample.type, sample.polarity, sample.tense, sample.certainty)
            combination_counts[combination] += 1
        
        # 빈도가 낮은 조합의 샘플들을 증강 후보로 선택
        rare_combinations = [combo for combo, count in combination_counts.items() if count < 5]
        
        for sample in self.samples:
            combination = (sample.type, sample.polarity, sample.tense, sample.certainty)
            if combination in rare_combinations:
                candidates.append(sample)
        
        logger.info(f"증강 후보: {len(candidates)}개")
        return candidates
    
    def generate_few_shot_examples(self, samples: List[Sample], error_patterns: List[str] = None) -> List[str]:
        """Few-shot 예시 생성"""
        if not samples:
            samples = self.samples
        
        # 오류 패턴이 있으면 해당 패턴의 정답 예시 우선 선택
        if error_patterns:
            targeted_samples = self._get_targeted_samples(samples, error_patterns)
        else:
            targeted_samples = samples
        
        # 다양성을 위해 각 속성별로 균형있게 선택
        balanced_samples = self._select_balanced_samples(targeted_samples, max_count=6)
        
        # Few-shot 형식으로 변환
        examples = []
        for i, sample in enumerate(balanced_samples, 1):
            example = f"""예시 {i}:
문장: "{sample.sentence}"
분류 결과:
- 유형: {sample.type}
- 극성: {sample.polarity}
- 시제: {sample.tense}
- 확실성: {sample.certainty}"""
            examples.append(example)
        
        return examples
    
    def _get_targeted_samples(self, samples: List[Sample], error_patterns: List[str]) -> List[Sample]:
        """오류 패턴에 맞는 타겟 샘플 선택"""
        targeted = []
        
        # 오류 패턴별로 정답 예시 찾기
        for pattern in error_patterns:
            # 패턴에 해당하는 속성과 값 추출
            if '_' in pattern:
                parts = pattern.split('_')
                if len(parts) >= 2:
                    attr_name = parts[0]
                    attr_value = parts[1]
                    
                    # 해당 속성값을 가진 샘플들 찾기
                    matching_samples = [
                        s for s in samples 
                        if hasattr(s, attr_name) and getattr(s, attr_name) == attr_value
                    ]
                    
                    if matching_samples:
                        targeted.extend(matching_samples[:2])  # 최대 2개씩
        
        return targeted if targeted else samples
    
    def _select_balanced_samples(self, samples: List[Sample], max_count: int = 6) -> List[Sample]:
        """균형잡힌 샘플 선택"""
        if len(samples) <= max_count:
            return samples
        
        # 각 속성별로 다양성 확보
        selected = []
        used_combinations = set()
        
        # 셔플해서 무작위성 추가
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        for sample in shuffled_samples:
            if len(selected) >= max_count:
                break
                
            # 속성 조합 확인
            combination = (sample.type, sample.polarity, sample.tense, sample.certainty)
            
            # 새로운 조합이면 선택
            if combination not in used_combinations:
                selected.append(sample)
                used_combinations.add(combination)
        
        # 부족하면 나머지로 채우기
        if len(selected) < max_count:
            remaining = [s for s in shuffled_samples if s not in selected]
            selected.extend(remaining[:max_count - len(selected)])
        
        return selected
    
    def augment_data_from_samples(self, base_samples: List[Sample]) -> List[Sample]:
        """샘플 기반 데이터 증강"""
        augmented = []
        
        for sample in base_samples:
            # 원본 유지
            augmented.append(sample)
            
            # 변형 버전 생성
            variations = self._create_variations(sample)
            augmented.extend(variations)
        
        logger.info(f"데이터 증강 완료: {len(base_samples)} → {len(augmented)}")
        return augmented
    
    def _create_variations(self, sample: Sample) -> List[Sample]:
        """샘플 변형 생성"""
        variations = []
        
        # 동의어 치환
        synonym_variation = self._apply_synonym_substitution(sample)
        if synonym_variation:
            variations.append(synonym_variation)
        
        # 문장 구조 변경
        structure_variation = self._apply_structure_change(sample)
        if structure_variation:
            variations.append(structure_variation)
        
        return variations
    
    def _apply_synonym_substitution(self, sample: Sample) -> Optional[Sample]:
        """동의어 치환"""
        # 간단한 동의어 사전
        synonyms = {
            '좋다': '훌륭하다',
            '나쁘다': '안좋다',
            '크다': '거대하다',
            '작다': '작은',
            '빠르다': '신속하다',
            '느리다': '더디다'
        }
        
        new_sentence = sample.sentence
        for original, synonym in synonyms.items():
            if original in new_sentence:
                new_sentence = new_sentence.replace(original, synonym, 1)
                break
        
        if new_sentence != sample.sentence:
            return Sample(
                sentence=new_sentence,
                type=sample.type,
                polarity=sample.polarity,
                tense=sample.tense,
                certainty=sample.certainty
            )
        
        return None
    
    def _apply_structure_change(self, sample: Sample) -> Optional[Sample]:
        """문장 구조 변경"""
        # 간단한 구조 변경 (실제로는 더 정교한 NLP 기법 필요)
        sentence = sample.sentence
        
        # "~이다" → "~입니다" 변경
        if sentence.endswith('이다.'):
            new_sentence = sentence[:-3] + '입니다.'
            return Sample(
                sentence=new_sentence,
                type=sample.type,
                polarity=sample.polarity,
                tense=sample.tense,
                certainty=sample.certainty
            )
        
        return None
    
    def validate_data_source_compliance(self, generated_content: str) -> bool:
        """데이터 소스 준수 여부 확인"""
        # samples.csv에서 나온 내용인지 확인
        if not self.samples:
            return False
        
        # 생성된 내용이 원본 샘플들과 유사한지 확인
        sample_sentences = [s.sentence for s in self.samples]
        
        # 간단한 유사도 검사 (실제로는 더 정교한 방법 필요)
        for sentence in sample_sentences:
            if sentence in generated_content:
                return True
        
        return False
    
    def get_analysis_report(self) -> str:
        """분석 리포트 생성"""
        if not self.analysis_result:
            return "분석이 수행되지 않았습니다."
        
        analysis = self.analysis_result
        
        report = f"""# 샘플 데이터 분석 리포트

## 기본 정보
- 총 샘플 수: {analysis.total_samples}개

## 속성별 분포
"""
        
        for attr, distribution in analysis.attribute_distributions.items():
            report += f"\n### {attr.upper()}\n"
            for value, count in distribution.items():
                percentage = (count / analysis.total_samples) * 100
                report += f"- {value}: {count}개 ({percentage:.1f}%)\n"
        
        report += f"""
## 패턴 분석
- 총 패턴 수: {len(analysis.pattern_examples)}개
- 경계 사례: {len(analysis.boundary_cases)}개
- 증강 후보: {len(analysis.augmentation_candidates)}개

## 대표 샘플
"""
        
        for attr, samples in analysis.representative_samples.items():
            report += f"\n### {attr.upper()} 대표 샘플\n"
            for sample in samples[:3]:  # 최대 3개만 표시
                report += f"- {sample.sentence}\n"
        
        return report