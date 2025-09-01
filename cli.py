"""
Command Line Interface for Gemini Prompt Optimizer
"""
import argparse
import sys
import os
from typing import Optional
import json
from datetime import datetime

from gemini_optimizer import GeminiPromptOptimizer
from config import OptimizationConfig
from utils.monitoring import OptimizationMonitor, RealtimeMonitor, create_final_report
from utils.logging_utils import setup_logging
from models.exceptions import GeminiOptimizerError

def create_config_file(output_path: str = ".env") -> None:
    """Create configuration file template"""
    config_template = """# Gemini API Configuration
GEMINI_API_KEY=your_gemini_api_key_here

# Optimization Parameters
TARGET_ACCURACY=0.95
MAX_ITERATIONS=10
BATCH_SIZE=50
API_RETRY_COUNT=3

# File Paths
SAMPLES_CSV_PATH=data/samples.csv
PROMPT_DIR=prompt
ANALYSIS_DIR=analysis

# Convergence Settings
CONVERGENCE_THRESHOLD=0.001
PATIENCE=3
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(config_template)
    
    print(f"Configuration template created at: {output_path}")
    print("Please edit the file and add your Gemini API key.")

def validate_files(csv_path: str, prompt_path: str) -> bool:
    """Validate required files exist"""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        return False
    
    if not os.path.exists(prompt_path):
        print(f"Error: Initial prompt file not found: {prompt_path}")
        return False
    
    return True

def print_system_info():
    """Print system information"""
    print("🤖 Gemini Prompt Optimizer")
    print("=" * 50)
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

def run_optimization_with_monitoring(config: OptimizationConfig, initial_prompt: str, 
                                   enable_realtime: bool = True, export_results: bool = True) -> int:
    """Run optimization with monitoring"""
    try:
        # Setup monitoring
        monitor = OptimizationMonitor(config.analysis_dir)
        realtime_monitor = RealtimeMonitor() if enable_realtime else None
        
        # Create optimizer
        optimizer = GeminiPromptOptimizer(config)
        
        print("🚀 Starting optimization process...")
        print(f"Target Accuracy: {config.target_accuracy:.2%}")
        print(f"Max Iterations: {config.max_iterations}")
        print(f"Dataset: {config.samples_csv_path}")
        print()
        
        # Start monitoring
        monitor.start_monitoring()
        
        # Run optimization
        result = optimizer.run_optimization(initial_prompt)
        
        # Display results
        print("\n" + "="*60)
        print("🎉 OPTIMIZATION COMPLETED!")
        print("="*60)
        print(result.get_final_report())
        
        # Create comprehensive final report
        final_report = create_final_report(result, monitor)
        
        # Save final report
        report_path = os.path.join(config.analysis_dir, "final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(final_report)
        
        print(f"\n📊 Final report saved to: {report_path}")
        
        # Export results if requested
        if export_results:
            export_dir = optimizer.export_results()
            monitor.export_metrics()
            print(f"📁 Results exported to: {export_dir}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⚠️  Optimization interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return 1

def test_api_connection(config: OptimizationConfig) -> bool:
    """Test API connection"""
    try:
        from services.gemini_client import GeminiClient
        
        print("🔍 Testing API connection...")
        client = GeminiClient(config)
        test_results = client.test_connection()
        
        print(f"API Key Valid: {'✅' if test_results['api_key_valid'] else '❌'}")
        print(f"Flash Model Available: {'✅' if test_results['flash_model_available'] else '❌'}")
        print(f"Pro Model Available: {'✅' if test_results['pro_model_available'] else '❌'}")
        print(f"Test Generation: {'✅' if test_results['test_generation_successful'] else '❌'}")
        
        if test_results.get('error_message'):
            print(f"Error: {test_results['error_message']}")
        
        return all([
            test_results['api_key_valid'],
            test_results['flash_model_available'],
            test_results['pro_model_available']
        ])
        
    except Exception as e:
        print(f"❌ API connection test failed: {e}")
        return False

def analyze_dataset(csv_path: str) -> None:
    """Analyze dataset and show statistics"""
    try:
        from services.csv_processor import CSVProcessor
        
        print(f"📊 Analyzing dataset: {csv_path}")
        processor = CSVProcessor(csv_path)
        
        # Load and validate
        samples = processor.load_samples()
        is_valid, errors = processor.validate_dataset()
        
        # Get statistics
        stats = processor.get_statistics()
        
        print(f"\n📈 Dataset Statistics:")
        print(f"Total Samples: {stats['total_samples']}")
        print(f"Valid Dataset: {'✅' if is_valid else '❌'}")
        
        if errors:
            print(f"Validation Errors: {len(errors)}")
            for error in errors[:3]:  # Show first 3 errors
                print(f"  - {error}")
            if len(errors) > 3:
                print(f"  ... and {len(errors) - 3} more errors")
        
        print(f"\n📊 Distribution:")
        for category, distribution in stats.items():
            if category.endswith('_distribution') and isinstance(distribution, dict):
                category_name = category.replace('_distribution', '').title()
                print(f"{category_name}:")
                for key, count in distribution.items():
                    percentage = (count / stats['total_samples']) * 100
                    print(f"  {key}: {count} ({percentage:.1f}%)")
                print()
        
        # Sentence length stats
        length_stats = stats.get('sentence_length_stats', {})
        if length_stats:
            print(f"Sentence Length:")
            print(f"  Min: {length_stats.get('min', 0)} chars")
            print(f"  Max: {length_stats.get('max', 0)} chars")
            print(f"  Average: {length_stats.get('avg', 0):.1f} chars")
            print(f"  Median: {length_stats.get('median', 0)} chars")
        
    except Exception as e:
        print(f"❌ Dataset analysis failed: {e}")

def create_sample_prompt(output_path: str) -> None:
    """Create sample prompt file"""
    sample_prompt = """당신은 한국어 문장 분류 전문가입니다.
주어진 문장을 다음 4가지 속성으로 분류해주세요.

분류 기준:

1. 유형 (Type):
   - 사실형: 객관적 사실, 통계, 뉴스 보도, 역사적 사건
   - 추론형: 개인의 분석, 의견, 해석, 추측, 가능성 표현
   - 대화형: 직접 인용문, 구어체 표현, 인사말, 대화문
   - 예측형: 미래 예측, 계획, 전망, 날씨 예보

2. 극성 (Polarity):
   - 긍정: 긍정적 내용 또는 중립적 서술
   - 부정: 부정적 내용, 문제점 지적, 실패, 거부
   - 미정: 질문문, 불확실한 추측, 가정법

3. 시제 (Tense):
   - 과거: 과거 시제 표현 ("~했다", "~였다")
   - 현재: 현재 시제, 일반적 사실 ("~이다", "~한다")
   - 미래: 미래 시제, 계획 ("~할 것이다", "~예정")

4. 확실성 (Certainty):
   - 확실: 명확하고 확정적인 내용
   - 불확실: 추측, 가능성 표현 ("~것 같다", "~할 수도")

출력 형식:
각 문장에 대해 "번호. 유형,극성,시제,확실성" 형식으로 답하세요.
쉼표로만 구분하고 추가 설명은 하지 마세요.

예시:
1. 사실형,긍정,과거,확실
2. 예측형,미정,미래,불확실
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(sample_prompt)
    
    print(f"Sample prompt created at: {output_path}")

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Gemini Prompt Optimizer - Automated prompt optimization for Korean sentence classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic optimization
  python cli.py optimize --initial-prompt prompt.txt

  # Custom settings
  python cli.py optimize --initial-prompt prompt.txt --target-accuracy 0.98 --max-iterations 15

  # Test API connection
  python cli.py test-api

  # Analyze dataset
  python cli.py analyze-dataset --csv-path data/samples.csv

  # Create configuration template
  python cli.py create-config

  # Create sample prompt
  python cli.py create-prompt --output prompt.txt
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Optimize command
    optimize_parser = subparsers.add_parser('optimize', help='Run prompt optimization')
    optimize_parser.add_argument('--initial-prompt', required=True, 
                                help='Path to initial system prompt file')
    optimize_parser.add_argument('--target-accuracy', type=float, default=0.95,
                                help='Target accuracy (0.0-1.0, default: 0.95)')
    optimize_parser.add_argument('--max-iterations', type=int, default=10,
                                help='Maximum iterations (default: 10)')
    optimize_parser.add_argument('--batch-size', type=int, default=50,
                                help='Batch size for classification (default: 50)')
    optimize_parser.add_argument('--csv-path', default='data/samples.csv',
                                help='Path to samples CSV file (default: data/samples.csv)')
    optimize_parser.add_argument('--output-dir', 
                                help='Output directory for results')
    optimize_parser.add_argument('--no-realtime', action='store_true',
                                help='Disable real-time monitoring display')
    optimize_parser.add_argument('--no-export', action='store_true',
                                help='Skip exporting detailed results')
    optimize_parser.add_argument('--log-level', default='INFO',
                                choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                                help='Logging level (default: INFO)')
    
    # Test API command
    test_parser = subparsers.add_parser('test-api', help='Test Gemini API connection')
    
    # Analyze dataset command
    analyze_parser = subparsers.add_parser('analyze-dataset', help='Analyze dataset statistics')
    analyze_parser.add_argument('--csv-path', default='data/samples.csv',
                               help='Path to CSV file (default: data/samples.csv)')
    
    # Create config command
    config_parser = subparsers.add_parser('create-config', help='Create configuration template')
    config_parser.add_argument('--output', default='.env',
                              help='Output path for config file (default: .env)')
    
    # Create prompt command
    prompt_parser = subparsers.add_parser('create-prompt', help='Create sample prompt template')
    prompt_parser.add_argument('--output', default='sample_prompt.txt',
                              help='Output path for prompt file (default: sample_prompt.txt)')
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Print system info
    print_system_info()
    
    # Handle commands
    if args.command == 'create-config':
        create_config_file(args.output)
        return 0
    
    elif args.command == 'create-prompt':
        create_sample_prompt(args.output)
        return 0
    
    elif args.command == 'analyze-dataset':
        analyze_dataset(args.csv_path)
        return 0
    
    elif args.command == 'test-api':
        try:
            config = OptimizationConfig.from_env()
            success = test_api_connection(config)
            return 0 if success else 1
        except Exception as e:
            print(f"❌ Configuration error: {e}")
            print("💡 Try running 'python cli.py create-config' first")
            return 1
    
    elif args.command == 'optimize':
        try:
            # Create configuration
            config = OptimizationConfig.from_env()
            
            # Override with command line arguments
            config.target_accuracy = args.target_accuracy
            config.max_iterations = args.max_iterations
            config.batch_size = args.batch_size
            config.samples_csv_path = args.csv_path
            
            if args.output_dir:
                config.analysis_dir = args.output_dir
                config.prompt_dir = os.path.join(args.output_dir, 'prompts')
            
            # Validate configuration
            config.validate()
            
            # Validate files
            if not validate_files(config.samples_csv_path, args.initial_prompt):
                return 1
            
            # Test API connection first
            if not test_api_connection(config):
                print("❌ API connection failed. Please check your configuration.")
                return 1
            
            print("✅ API connection successful!")
            print()
            
            # Run optimization
            return run_optimization_with_monitoring(
                config, 
                args.initial_prompt,
                enable_realtime=not args.no_realtime,
                export_results=not args.no_export
            )
            
        except FileNotFoundError as e:
            print(f"❌ File not found: {e}")
            print("💡 Make sure all required files exist")
            return 1
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
            return 1
        except GeminiOptimizerError as e:
            print(f"❌ Optimization error: {e}")
            return 1
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return 1
    
    else:
        parser.print_help()
        return 1

if __name__ == '__main__':
    sys.exit(main())