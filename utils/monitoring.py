"""
Advanced monitoring and reporting utilities for Gemini Prompt Optimizer
"""
import os
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from dataclasses import asdict
from models.data_models import IterationState, OptimizationResult
from utils.file_utils import write_json_file, read_json_file, ensure_directory_exists

logger = logging.getLogger("gemini_optimizer.monitoring")

class OptimizationMonitor:
    """Advanced monitoring system for optimization process"""
    
    def __init__(self, output_dir: str = "monitoring"):
        self.output_dir = output_dir
        self.start_time = None
        self.metrics_history = []
        self.performance_data = {}
        self.alerts = []
        self.iteration_summaries = []  # Store iteration summaries for next steps
        self.improvement_history = []  # Track specific improvements
        self.visualization_data = {}   # Store data for graph generation
        
        ensure_directory_exists(output_dir)
        logger.info(f"OptimizationMonitor initialized with output dir: {output_dir}")
    
    def start_monitoring(self) -> None:
        """Start monitoring session"""
        self.start_time = time.time()
        self.metrics_history = []
        self.performance_data = {}
        self.alerts = []
        self.iteration_summaries = []
        self.improvement_history = []
        self.visualization_data = {}
        
        logger.info("Monitoring session started")
    
    def record_iteration_metrics(self, iteration: int, state: IterationState, 
                                additional_metrics: Dict[str, Any] = None) -> None:
        """Record metrics for current iteration"""
        timestamp = time.time()
        
        metrics = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).isoformat(),
            "iteration": iteration,
            "accuracy": state.current_accuracy,
            "best_accuracy": state.best_accuracy,
            "target_accuracy": state.target_accuracy,
            "error_count": state.error_count,
            "correct_predictions": state.correct_predictions,
            "total_samples": state.total_samples,
            "is_converged": state.is_converged,
            "elapsed_time": timestamp - self.start_time if self.start_time else 0
        }
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.metrics_history.append(metrics)
        
        # Track improvements
        if len(self.metrics_history) > 1:
            prev_accuracy = self.metrics_history[-2]["accuracy"]
            current_accuracy = state.current_accuracy
            if current_accuracy > prev_accuracy:
                improvement = {
                    "iteration": iteration,
                    "improvement": current_accuracy - prev_accuracy,
                    "timestamp": timestamp,
                    "type": "accuracy_increase"
                }
                self.improvement_history.append(improvement)
        
        # Check for alerts
        self._check_alerts(metrics)
        
        logger.debug(f"Recorded metrics for iteration {iteration}")
    
    def record_performance_data(self, category: str, data: Dict[str, Any]) -> None:
        """Record performance data by category"""
        if category not in self.performance_data:
            self.performance_data[category] = []
        
        data_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            **data
        }
        
        self.performance_data[category].append(data_entry)
        logger.debug(f"Recorded performance data for category: {category}")
    
    def _check_alerts(self, metrics: Dict[str, Any]) -> None:
        """Check for alert conditions"""
        alerts = []
        
        # Alert: No improvement for multiple iterations
        if len(self.metrics_history) >= 3:
            recent_accuracies = [m["accuracy"] for m in self.metrics_history[-3:]]
            if all(acc == recent_accuracies[0] for acc in recent_accuracies):
                alerts.append({
                    "type": "no_improvement",
                    "message": "No accuracy improvement for 3 consecutive iterations",
                    "severity": "warning",
                    "timestamp": time.time()
                })
        
        # Alert: Accuracy degradation
        if len(self.metrics_history) >= 2:
            current_acc = metrics["accuracy"]
            previous_acc = self.metrics_history[-2]["accuracy"]
            if current_acc < previous_acc - 0.05:  # 5% degradation
                alerts.append({
                    "type": "accuracy_degradation",
                    "message": f"Accuracy dropped by {(previous_acc - current_acc):.3f}",
                    "severity": "warning",
                    "timestamp": time.time()
                })
        
        # Alert: Long execution time
        elapsed_time = metrics.get("elapsed_time", 0)
        if elapsed_time > 3600:  # 1 hour
            alerts.append({
                "type": "long_execution",
                "message": f"Optimization running for {elapsed_time/3600:.1f} hours",
                "severity": "info",
                "timestamp": time.time()
            })
        
        # Add alerts to history
        self.alerts.extend(alerts)
        
        # Log alerts
        for alert in alerts:
            if alert["severity"] == "warning":
                logger.warning(f"Alert: {alert['message']}")
            else:
                logger.info(f"Alert: {alert['message']}")
    
    def generate_progress_report(self) -> str:
        """Generate detailed progress report"""
        if not self.metrics_history:
            return "No metrics data available"
        
        latest = self.metrics_history[-1]
        initial = self.metrics_history[0]
        
        # Calculate statistics
        accuracies = [m["accuracy"] for m in self.metrics_history]
        best_accuracy = max(accuracies)
        worst_accuracy = min(accuracies)
        avg_accuracy = sum(accuracies) / len(accuracies)
        
        # Calculate improvement
        total_improvement = latest["accuracy"] - initial["accuracy"]
        improvement_rate = total_improvement / len(self.metrics_history) if len(self.metrics_history) > 1 else 0
        
        # Format elapsed time
        elapsed_time = latest.get("elapsed_time", 0)
        elapsed_str = str(timedelta(seconds=int(elapsed_time)))
        
        report = f"""
{'='*60}
OPTIMIZATION PROGRESS REPORT
{'='*60}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

CURRENT STATUS:
- Iteration: {latest['iteration']}
- Current Accuracy: {latest['accuracy']:.4f}
- Best Accuracy: {best_accuracy:.4f}
- Errors: {latest['error_count']}/{latest['total_samples']}
- Converged: {latest['is_converged']}

PERFORMANCE STATISTICS:
- Initial Accuracy: {initial['accuracy']:.4f}
- Best Accuracy: {best_accuracy:.4f}
- Worst Accuracy: {worst_accuracy:.4f}
- Average Accuracy: {avg_accuracy:.4f}
- Total Improvement: {total_improvement:+.4f}
- Improvement Rate: {improvement_rate:+.4f} per iteration

TIMING:
- Elapsed Time: {elapsed_str}
- Average Time per Iteration: {elapsed_time/len(self.metrics_history):.1f} seconds

ALERTS:
"""
        
        if self.alerts:
            recent_alerts = [a for a in self.alerts if time.time() - a["timestamp"] < 3600]  # Last hour
            for alert in recent_alerts[-5:]:  # Show last 5 alerts
                alert_time = datetime.fromtimestamp(alert["timestamp"]).strftime('%H:%M:%S')
                report += f"- [{alert_time}] {alert['severity'].upper()}: {alert['message']}\n"
        else:
            report += "- No alerts\n"
        
        report += f"\n{'='*60}"
        
        return report
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimization_metrics_{timestamp}.json"
        
        filepath = os.path.join(self.output_dir, filename)
        
        export_data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_iterations": len(self.metrics_history),
                "start_time": datetime.fromtimestamp(self.start_time).isoformat() if self.start_time else None,
                "total_duration": time.time() - self.start_time if self.start_time else 0
            },
            "metrics_history": self.metrics_history,
            "performance_data": self.performance_data,
            "alerts": self.alerts
        }
        
        write_json_file(filepath, export_data)
        logger.info(f"Metrics exported to {filepath}")
        
        return filepath
    
    def display_iteration_start(self, iteration: int, target_accuracy: float, 
                               current_best: float, strategy: str = None) -> None:
        """Display current goals and progress when iteration starts (Requirement 9.1)"""
        print("\n" + "="*80)
        print(f"🚀 ITERATION {iteration} STARTING")
        print("="*80)
        print(f"📊 Current Status:")
        print(f"   Target Accuracy: {target_accuracy:.1%}")
        print(f"   Current Best: {current_best:.1%}")
        print(f"   Gap to Target: {(target_accuracy - current_best):.1%}")
        
        if strategy:
            print(f"   Strategy: {strategy}")
        
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        print(f"   Elapsed Time: {elapsed_time/60:.1f} minutes")
        
        # Show progress bar
        progress = min(1.0, current_best / target_accuracy) if target_accuracy > 0 else 0
        bar_length = 40
        filled_length = int(bar_length * progress)
        bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"   Progress: [{bar}] {progress:.1%}")
        
        print("="*80)
        logger.info(f"Iteration {iteration} started - Target: {target_accuracy:.1%}, Best: {current_best:.1%}")
    
    def generate_performance_visualization(self) -> Dict[str, Any]:
        """Generate performance trend visualization data (Requirement 9.2)"""
        if not self.metrics_history:
            return {}
        
        # Prepare data for visualization
        iterations = [m["iteration"] for m in self.metrics_history]
        accuracies = [m["accuracy"] for m in self.metrics_history]
        best_accuracies = [m["best_accuracy"] for m in self.metrics_history]
        error_counts = [m["error_count"] for m in self.metrics_history]
        
        # Calculate moving averages
        window_size = min(3, len(accuracies))
        moving_avg = []
        for i in range(len(accuracies)):
            start_idx = max(0, i - window_size + 1)
            avg = sum(accuracies[start_idx:i+1]) / (i - start_idx + 1)
            moving_avg.append(avg)
        
        # Calculate improvement rate
        improvement_rates = [0]  # First iteration has no improvement rate
        for i in range(1, len(accuracies)):
            rate = accuracies[i] - accuracies[i-1]
            improvement_rates.append(rate)
        
        visualization_data = {
            "performance_trend": {
                "iterations": iterations,
                "accuracies": accuracies,
                "best_accuracies": best_accuracies,
                "moving_average": moving_avg,
                "improvement_rates": improvement_rates,
                "error_counts": error_counts
            },
            "statistics": {
                "total_iterations": len(iterations),
                "best_accuracy": max(accuracies) if accuracies else 0,
                "worst_accuracy": min(accuracies) if accuracies else 0,
                "average_accuracy": sum(accuracies) / len(accuracies) if accuracies else 0,
                "total_improvement": accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0,
                "volatility": self._calculate_volatility(accuracies)
            }
        }
        
        # Store for later use
        self.visualization_data = visualization_data
        
        # Generate ASCII chart for console display
        self._display_ascii_chart(accuracies, iterations)
        
        return visualization_data
    
    def _display_ascii_chart(self, accuracies: List[float], iterations: List[int]) -> None:
        """Display ASCII chart of performance trends"""
        if not accuracies:
            return
        
        print("\n📈 PERFORMANCE TREND VISUALIZATION")
        print("-" * 60)
        
        # Normalize accuracies for display (0-20 scale)
        min_acc = min(accuracies)
        max_acc = max(accuracies)
        acc_range = max_acc - min_acc if max_acc > min_acc else 0.1
        
        chart_height = 15
        chart_width = min(50, len(accuracies) * 2)
        
        # Create chart
        for row in range(chart_height, -1, -1):
            line = f"{min_acc + (row/chart_height) * acc_range:.3f} |"
            
            for i, acc in enumerate(accuracies):
                if i * 2 >= chart_width:
                    break
                    
                normalized_acc = (acc - min_acc) / acc_range
                if normalized_acc * chart_height >= row - 0.5:
                    line += "██"
                else:
                    line += "  "
            
            print(line)
        
        # X-axis
        x_axis = "      +"
        x_labels = "       "
        
        for i in range(0, min(len(iterations), chart_width // 2)):
            x_axis += "--"
            if i % 5 == 0:  # Show every 5th iteration
                x_labels += f"{iterations[i]:2d}"
            else:
                x_labels += "  "
        
        print(x_axis)
        print(x_labels)
        print(f"       Iterations (Latest: {accuracies[-1]:.1%})")
        print("-" * 60)
    
    def summarize_improvements_and_next_steps(self, analysis_result: Dict[str, Any] = None) -> str:
        """Summarize key improvements and suggest next steps (Requirement 9.3)"""
        if not self.metrics_history:
            return "No data available for analysis"
        
        latest = self.metrics_history[-1]
        
        # Analyze recent performance
        recent_window = min(3, len(self.metrics_history))
        recent_metrics = self.metrics_history[-recent_window:]
        recent_accuracies = [m["accuracy"] for m in recent_metrics]
        
        # Calculate improvements
        improvements = []
        if len(self.metrics_history) > 1:
            total_improvement = latest["accuracy"] - self.metrics_history[0]["accuracy"]
            if total_improvement > 0:
                improvements.append(f"Overall accuracy improved by {total_improvement:.1%}")
            
            # Recent trend
            if len(recent_accuracies) >= 2:
                recent_improvement = recent_accuracies[-1] - recent_accuracies[0]
                if recent_improvement > 0.01:
                    improvements.append(f"Recent upward trend (+{recent_improvement:.1%})")
                elif recent_improvement < -0.01:
                    improvements.append(f"Recent downward trend ({recent_improvement:.1%})")
                else:
                    improvements.append("Performance stabilizing")
        
        # Identify next steps based on current state
        next_steps = []
        
        # Check convergence
        if self._calculate_trend(recent_accuracies) == "stable":
            next_steps.append("Consider trying different optimization strategy")
            next_steps.append("Analyze error patterns for targeted improvements")
        elif self._calculate_trend(recent_accuracies) == "improving":
            next_steps.append("Continue current optimization approach")
            next_steps.append("Monitor for potential overfitting")
        else:
            next_steps.append("Review recent changes that may have caused decline")
            next_steps.append("Consider reverting to previous best configuration")
        
        # Add analysis-specific suggestions
        if analysis_result:
            error_patterns = analysis_result.get("error_patterns", {})
            if error_patterns:
                top_error = max(error_patterns.items(), key=lambda x: x[1])
                next_steps.append(f"Focus on reducing '{top_error[0]}' errors ({top_error[1]} occurrences)")
        
        # Target-based suggestions
        target_accuracy = latest.get("target_accuracy", 0.95)
        current_accuracy = latest["accuracy"]
        gap = target_accuracy - current_accuracy
        
        if gap > 0.1:
            next_steps.append("Large accuracy gap - consider fundamental prompt restructuring")
        elif gap > 0.05:
            next_steps.append("Moderate gap - fine-tune classification rules")
        elif gap > 0.01:
            next_steps.append("Small gap - focus on edge cases and boundary conditions")
        else:
            next_steps.append("Target achieved - validate with additional test sets")
        
        # Create summary
        summary = f"""
🔍 ANALYSIS SUMMARY - Iteration {latest['iteration']}
{'='*60}

📊 KEY IMPROVEMENTS:
"""
        
        if improvements:
            for improvement in improvements:
                summary += f"   ✅ {improvement}\n"
        else:
            summary += "   ⚠️  No significant improvements detected\n"
        
        summary += f"""
🎯 NEXT STEPS:
"""
        for i, step in enumerate(next_steps, 1):
            summary += f"   {i}. {step}\n"
        
        summary += f"""
📈 CURRENT METRICS:
   • Accuracy: {current_accuracy:.1%} (Target: {target_accuracy:.1%})
   • Errors: {latest['error_count']}/{latest['total_samples']}
   • Trend: {self._calculate_trend(recent_accuracies)}
   • Volatility: {self._calculate_volatility(recent_accuracies):.3f}

{'='*60}
"""
        
        # Store summary for history
        summary_data = {
            "iteration": latest['iteration'],
            "timestamp": time.time(),
            "improvements": improvements,
            "next_steps": next_steps,
            "metrics": {
                "accuracy": current_accuracy,
                "target_accuracy": target_accuracy,
                "gap": gap,
                "trend": self._calculate_trend(recent_accuracies)
            }
        }
        self.iteration_summaries.append(summary_data)
        
        print(summary)
        logger.info(f"Analysis summary generated for iteration {latest['iteration']}")
        
        return summary
    
    def generate_complete_performance_history(self) -> str:
        """Generate complete performance improvement history (Requirement 9.4)"""
        if not self.metrics_history:
            return "No performance history available"
        
        # Calculate comprehensive statistics
        accuracies = [m["accuracy"] for m in self.metrics_history]
        iterations = [m["iteration"] for m in self.metrics_history]
        
        initial_accuracy = accuracies[0]
        final_accuracy = accuracies[-1]
        best_accuracy = max(accuracies)
        best_iteration = iterations[accuracies.index(best_accuracy)]
        
        total_improvement = final_accuracy - initial_accuracy
        best_improvement = best_accuracy - initial_accuracy
        
        # Find significant milestones
        milestones = []
        accuracy_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
        
        for threshold in accuracy_thresholds:
            for i, acc in enumerate(accuracies):
                if acc >= threshold:
                    milestones.append({
                        "threshold": threshold,
                        "iteration": iterations[i],
                        "accuracy": acc
                    })
                    break
        
        # Calculate improvement phases
        phases = []
        window_size = max(1, len(accuracies) // 5)  # Divide into 5 phases
        
        for i in range(0, len(accuracies), window_size):
            end_idx = min(i + window_size, len(accuracies))
            phase_accuracies = accuracies[i:end_idx]
            phase_iterations = iterations[i:end_idx]
            
            if phase_accuracies:
                phases.append({
                    "phase": len(phases) + 1,
                    "iterations": f"{phase_iterations[0]}-{phase_iterations[-1]}",
                    "start_accuracy": phase_accuracies[0],
                    "end_accuracy": phase_accuracies[-1],
                    "improvement": phase_accuracies[-1] - phase_accuracies[0],
                    "best_in_phase": max(phase_accuracies)
                })
        
        # Generate comprehensive report
        history_report = f"""
🏆 COMPLETE PERFORMANCE IMPROVEMENT HISTORY
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Duration: {(time.time() - self.start_time)/3600:.2f} hours

📊 OVERALL PERFORMANCE SUMMARY:
   Initial Accuracy:     {initial_accuracy:.1%}
   Final Accuracy:       {final_accuracy:.1%}
   Best Accuracy:        {best_accuracy:.1%} (Iteration {best_iteration})
   Total Improvement:    {total_improvement:+.1%}
   Best Improvement:     {best_improvement:+.1%}
   Total Iterations:     {len(iterations)}

🎯 MILESTONE ACHIEVEMENTS:
"""
        
        for milestone in milestones:
            history_report += f"   {milestone['threshold']:.0%} Accuracy: Reached at Iteration {milestone['iteration']} ({milestone['accuracy']:.1%})\n"
        
        if not milestones:
            history_report += "   No major milestones achieved\n"
        
        history_report += f"""
📈 IMPROVEMENT PHASES:
"""
        
        for phase in phases:
            trend_symbol = "📈" if phase["improvement"] > 0 else "📉" if phase["improvement"] < 0 else "➡️"
            history_report += f"   Phase {phase['phase']} (Iter {phase['iterations']}): {phase['start_accuracy']:.1%} → {phase['end_accuracy']:.1%} {trend_symbol} ({phase['improvement']:+.1%})\n"
        
        # Add detailed iteration history
        history_report += f"""
📋 DETAILED ITERATION HISTORY:
   Iter  Accuracy  Best    Errors  Trend   Notes
   ----  --------  ------  ------  -----   -----
"""
        
        for i, metrics in enumerate(self.metrics_history):
            # Calculate trend for this iteration
            if i == 0:
                trend = "START"
            else:
                prev_acc = self.metrics_history[i-1]["accuracy"]
                curr_acc = metrics["accuracy"]
                if curr_acc > prev_acc + 0.01:
                    trend = "UP"
                elif curr_acc < prev_acc - 0.01:
                    trend = "DOWN"
                else:
                    trend = "FLAT"
            
            # Add notes for significant events
            notes = []
            if metrics["accuracy"] == best_accuracy:
                notes.append("BEST")
            if i < len(self.iteration_summaries):
                summary = self.iteration_summaries[i]
                if summary.get("improvements"):
                    notes.append("IMPROVED")
            
            notes_str = ",".join(notes) if notes else ""
            
            history_report += f"   {metrics['iteration']:4d}  {metrics['accuracy']:7.1%}  {metrics['best_accuracy']:6.1%}  {metrics['error_count']:6d}  {trend:5s}   {notes_str}\n"
        
        # Add performance analytics
        volatility = self._calculate_volatility(accuracies)
        consistency = 1.0 - volatility
        convergence_speed = self._calculate_convergence_speed()
        efficiency = self._calculate_efficiency_score()
        
        history_report += f"""
🔬 PERFORMANCE ANALYTICS:
   Consistency Score:    {consistency:.3f} (1.0 = perfectly consistent)
   Volatility:           {volatility:.3f} (0.0 = no variation)
   Convergence Speed:    {convergence_speed:.3f} (higher = faster convergence)
   Efficiency Score:     {efficiency:.3f} (improvement per iteration)

🚨 ALERTS SUMMARY:
   Total Alerts:         {len(self.alerts)}
   Alert Types:          {', '.join(set(a['type'] for a in self.alerts)) if self.alerts else 'None'}

💡 OPTIMIZATION INSIGHTS:
"""
        
        # Generate insights based on performance patterns
        insights = []
        
        if total_improvement > 0.1:
            insights.append("Significant improvement achieved - optimization strategy was effective")
        elif total_improvement > 0.05:
            insights.append("Moderate improvement - consider more aggressive optimization")
        elif total_improvement > 0:
            insights.append("Minimal improvement - may need different approach")
        else:
            insights.append("No improvement or regression - review optimization strategy")
        
        if volatility < 0.02:
            insights.append("Low volatility indicates stable optimization process")
        elif volatility > 0.05:
            insights.append("High volatility suggests unstable optimization - consider regularization")
        
        if convergence_speed > 0.5:
            insights.append("Fast convergence - efficient optimization")
        elif convergence_speed < 0.1:
            insights.append("Slow convergence - may need more iterations or different strategy")
        
        for insight in insights:
            history_report += f"   • {insight}\n"
        
        history_report += f"\n{'='*80}\n"
        
        # Save detailed history to file
        history_file = os.path.join(self.output_dir, "complete_performance_history.txt")
        with open(history_file, 'w', encoding='utf-8') as f:
            f.write(history_report)
        
        logger.info(f"Complete performance history saved to {history_file}")
        
        return history_report
    
    def run_complete_monitoring_cycle(self, iteration: int, state: IterationState, 
                                    strategy: str = None, analysis_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Run complete monitoring cycle for an iteration (integrates all requirements)"""
        
        # Requirement 9.1: Display iteration start with goals and progress
        self.display_iteration_start(iteration, state.target_accuracy, state.best_accuracy, strategy)
        
        # Record metrics
        self.record_iteration_metrics(iteration, state)
        
        # Requirement 9.2: Generate performance visualization after test completion
        visualization_data = self.generate_performance_visualization()
        
        # Requirement 9.3: Summarize improvements and next steps after analysis
        summary = self.summarize_improvements_and_next_steps(analysis_result)
        
        # Return comprehensive monitoring data
        monitoring_result = {
            "iteration": iteration,
            "visualization_data": visualization_data,
            "summary": summary,
            "current_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "alerts": [a for a in self.alerts if time.time() - a["timestamp"] < 300],  # Last 5 minutes
            "recommendations": self._generate_recommendations(state)
        }
        
        return monitoring_result
    
    def _generate_recommendations(self, state: IterationState) -> List[str]:
        """Generate actionable recommendations based on current state"""
        recommendations = []
        
        # Accuracy-based recommendations
        gap = state.target_accuracy - state.current_accuracy
        if gap > 0.1:
            recommendations.append("Consider major prompt restructuring - large accuracy gap detected")
        elif gap > 0.05:
            recommendations.append("Fine-tune classification rules and add more examples")
        elif gap > 0.01:
            recommendations.append("Focus on edge cases and boundary conditions")
        
        # Error-based recommendations
        error_rate = state.error_count / state.total_samples if state.total_samples > 0 else 0
        if error_rate > 0.3:
            recommendations.append("High error rate - review fundamental prompt structure")
        elif error_rate > 0.1:
            recommendations.append("Moderate errors - analyze error patterns for targeted fixes")
        
        # Convergence-based recommendations
        if len(self.metrics_history) >= 3:
            recent_accuracies = [m["accuracy"] for m in self.metrics_history[-3:]]
            trend = self._calculate_trend(recent_accuracies)
            
            if trend == "stable" and not state.is_converged:
                recommendations.append("Performance plateaued - try different optimization strategy")
            elif trend == "declining":
                recommendations.append("Performance declining - consider reverting recent changes")
        
        # Time-based recommendations
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        if elapsed_time > 3600:  # 1 hour
            recommendations.append("Long optimization time - consider checkpointing progress")
        
        return recommendations
    
    def export_complete_monitoring_data(self) -> str:
        """Export all monitoring data including visualizations and history"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_dir = os.path.join(self.output_dir, f"monitoring_export_{timestamp}")
        ensure_directory_exists(export_dir)
        
        # Export metrics history
        metrics_file = os.path.join(export_dir, "metrics_history.json")
        write_json_file(metrics_file, {
            "metrics_history": self.metrics_history,
            "improvement_history": self.improvement_history,
            "iteration_summaries": self.iteration_summaries,
            "alerts": self.alerts
        })
        
        # Export visualization data
        if self.visualization_data:
            viz_file = os.path.join(export_dir, "visualization_data.json")
            write_json_file(viz_file, self.visualization_data)
        
        # Export complete performance history
        history_file = os.path.join(export_dir, "complete_history.txt")
        history_content = self.generate_complete_performance_history()
        with open(history_file, 'w', encoding='utf-8') as f:
            f.write(history_content)
        
        # Create summary dashboard
        dashboard_file = os.path.join(export_dir, "monitoring_dashboard.json")
        dashboard_data = self.create_summary_dashboard()
        write_json_file(dashboard_file, dashboard_data)
        
        logger.info(f"Complete monitoring data exported to {export_dir}")
        return export_dir
    
    def create_summary_dashboard(self) -> Dict[str, Any]:
        """Create summary dashboard data"""
        if not self.metrics_history:
            return {"error": "No metrics data available"}
        
        latest = self.metrics_history[-1]
        accuracies = [m["accuracy"] for m in self.metrics_history]
        
        dashboard = {
            "current_status": {
                "iteration": latest["iteration"],
                "accuracy": latest["accuracy"],
                "best_accuracy": max(accuracies),
                "error_count": latest["error_count"],
                "converged": latest["is_converged"],
                "elapsed_time": latest.get("elapsed_time", 0)
            },
            "performance_trend": {
                "accuracy_history": accuracies,
                "improvement_trend": self._calculate_trend(accuracies),
                "volatility": self._calculate_volatility(accuracies)
            },
            "alerts_summary": {
                "total_alerts": len(self.alerts),
                "recent_alerts": len([a for a in self.alerts if time.time() - a["timestamp"] < 3600]),
                "alert_types": list(set(a["type"] for a in self.alerts))
            },
            "performance_metrics": self._calculate_performance_metrics()
        }
        
        return dashboard
    
    def _calculate_trend(self, values: List[float], window: int = 3) -> str:
        """Calculate trend direction"""
        if len(values) < window:
            return "insufficient_data"
        
        recent_values = values[-window:]
        if len(set(recent_values)) == 1:
            return "stable"
        
        # Simple linear trend
        x = list(range(len(recent_values)))
        y = recent_values
        
        # Calculate slope
        n = len(x)
        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (n * sum(x[i]**2 for i in range(n)) - sum(x)**2)
        
        if slope > 0.001:
            return "improving"
        elif slope < -0.001:
            return "declining"
        else:
            return "stable"
    
    def _calculate_volatility(self, values: List[float]) -> float:
        """Calculate volatility (standard deviation)"""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate various performance metrics"""
        if not self.metrics_history:
            return {}
        
        accuracies = [m["accuracy"] for m in self.metrics_history]
        
        return {
            "mean_accuracy": sum(accuracies) / len(accuracies),
            "median_accuracy": sorted(accuracies)[len(accuracies) // 2],
            "accuracy_range": max(accuracies) - min(accuracies),
            "consistency_score": 1.0 - self._calculate_volatility(accuracies),
            "convergence_speed": self._calculate_convergence_speed(),
            "efficiency_score": self._calculate_efficiency_score()
        }
    
    def _calculate_convergence_speed(self) -> float:
        """Calculate how quickly the optimization converges"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        accuracies = [m["accuracy"] for m in self.metrics_history]
        initial_acc = accuracies[0]
        final_acc = accuracies[-1]
        
        if final_acc <= initial_acc:
            return 0.0
        
        # Find iteration where 80% of improvement was achieved
        target_improvement = (final_acc - initial_acc) * 0.8
        target_accuracy = initial_acc + target_improvement
        
        for i, acc in enumerate(accuracies):
            if acc >= target_accuracy:
                return 1.0 / (i + 1)  # Inverse of iterations needed
        
        return 1.0 / len(accuracies)
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate efficiency score based on improvement per iteration"""
        if len(self.metrics_history) < 2:
            return 0.0
        
        accuracies = [m["accuracy"] for m in self.metrics_history]
        total_improvement = accuracies[-1] - accuracies[0]
        iterations = len(accuracies)
        
        # Normalize by maximum possible improvement (assuming max accuracy is 1.0)
        max_possible_improvement = 1.0 - accuracies[0]
        
        if max_possible_improvement <= 0:
            return 1.0  # Already at maximum
        
        efficiency = (total_improvement / max_possible_improvement) / iterations
        return min(1.0, max(0.0, efficiency))

class RealtimeMonitor:
    """Real-time monitoring display for console output"""
    
    def __init__(self):
        self.last_update = 0
        self.update_interval = 5  # seconds
    
    def should_update(self) -> bool:
        """Check if display should be updated"""
        current_time = time.time()
        if current_time - self.last_update >= self.update_interval:
            self.last_update = current_time
            return True
        return False
    
    def display_progress(self, state: IterationState, additional_info: Dict[str, Any] = None) -> None:
        """Display real-time progress"""
        if not self.should_update():
            return
        
        # Clear screen (works on most terminals)
        os.system('cls' if os.name == 'nt' else 'clear')
        
        print("🔄 GEMINI PROMPT OPTIMIZER - REAL-TIME MONITORING")
        print("=" * 60)
        print(f"Iteration: {state.iteration}")
        print(f"Current Accuracy: {state.current_accuracy:.4f}")
        print(f"Best Accuracy: {state.best_accuracy:.4f}")
        print(f"Target Accuracy: {state.target_accuracy:.4f}")
        print(f"Progress: {(state.current_accuracy/state.target_accuracy)*100:.1f}%")
        print(f"Errors: {state.error_count}/{state.total_samples}")
        print(f"Converged: {'✅' if state.is_converged else '❌'}")
        
        if additional_info:
            print("\nAdditional Info:")
            for key, value in additional_info.items():
                print(f"  {key}: {value}")
        
        print("\n" + "=" * 60)
        print("Press Ctrl+C to stop optimization")

def create_final_report(result: OptimizationResult, monitor: OptimizationMonitor) -> str:
    """Create comprehensive final report"""
    dashboard = monitor.create_summary_dashboard()
    
    report = f"""
{'='*80}
GEMINI PROMPT OPTIMIZER - FINAL REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OPTIMIZATION RESULTS:
{result.get_final_report()}

PERFORMANCE ANALYSIS:
- Mean Accuracy: {dashboard.get('performance_metrics', {}).get('mean_accuracy', 0):.4f}
- Consistency Score: {dashboard.get('performance_metrics', {}).get('consistency_score', 0):.4f}
- Convergence Speed: {dashboard.get('performance_metrics', {}).get('convergence_speed', 0):.4f}
- Efficiency Score: {dashboard.get('performance_metrics', {}).get('efficiency_score', 0):.4f}

TREND ANALYSIS:
- Accuracy Trend: {dashboard.get('performance_trend', {}).get('improvement_trend', 'unknown')}
- Volatility: {dashboard.get('performance_trend', {}).get('volatility', 0):.4f}

ALERTS SUMMARY:
- Total Alerts: {dashboard.get('alerts_summary', {}).get('total_alerts', 0)}
- Alert Types: {', '.join(dashboard.get('alerts_summary', {}).get('alert_types', []))}

{'='*80}
"""
    
    return report