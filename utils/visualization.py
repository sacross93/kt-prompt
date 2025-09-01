"""
Visualization utilities for optimization results
"""
import os
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger("gemini_optimizer.visualization")

def create_progress_chart_data(iteration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create data for progress visualization"""
    if not iteration_history:
        return {}
    
    iterations = [h["iteration"] for h in iteration_history]
    accuracies = [h["accuracy"] for h in iteration_history]
    error_counts = [h["error_count"] for h in iteration_history]
    
    return {
        "chart_type": "line",
        "title": "Optimization Progress",
        "x_axis": {
            "label": "Iteration",
            "data": iterations
        },
        "y_axes": [
            {
                "label": "Accuracy",
                "data": accuracies,
                "color": "#2E8B57",
                "type": "line"
            },
            {
                "label": "Error Count",
                "data": error_counts,
                "color": "#DC143C",
                "type": "bar",
                "secondary": True
            }
        ]
    }

def create_accuracy_distribution_data(iteration_history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Create accuracy distribution visualization data"""
    if not iteration_history:
        return {}
    
    accuracies = [h["accuracy"] for h in iteration_history]
    
    # Create histogram bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    bin_counts = [0] * (len(bins) - 1)
    
    for accuracy in accuracies:
        for i in range(len(bins) - 1):
            if bins[i] <= accuracy < bins[i + 1]:
                bin_counts[i] += 1
                break
        else:
            if accuracy == 1.0:
                bin_counts[-1] += 1
    
    return {
        "chart_type": "histogram",
        "title": "Accuracy Distribution",
        "x_axis": {
            "label": "Accuracy Range",
            "data": [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
        },
        "y_axis": {
            "label": "Frequency",
            "data": bin_counts
        }
    }

def create_error_pattern_data(error_patterns: Dict[str, int]) -> Dict[str, Any]:
    """Create error pattern visualization data"""
    if not error_patterns:
        return {}
    
    # Sort by frequency
    sorted_patterns = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)
    
    # Take top 10 patterns
    top_patterns = sorted_patterns[:10]
    
    patterns = [pattern for pattern, count in top_patterns]
    counts = [count for pattern, count in top_patterns]
    
    return {
        "chart_type": "bar",
        "title": "Top Error Patterns",
        "x_axis": {
            "label": "Error Pattern",
            "data": patterns
        },
        "y_axis": {
            "label": "Frequency",
            "data": counts
        }
    }

def create_category_performance_data(category_accuracy: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Create category performance visualization data"""
    if not category_accuracy:
        return {}
    
    categories = list(category_accuracy.keys())
    accuracies = [category_accuracy[cat]["accuracy"] for cat in categories]
    
    # Translate category names to Korean
    category_names = {
        "type": "유형",
        "polarity": "극성", 
        "tense": "시제",
        "certainty": "확실성"
    }
    
    korean_categories = [category_names.get(cat, cat) for cat in categories]
    
    return {
        "chart_type": "radar",
        "title": "Category Performance",
        "categories": korean_categories,
        "data": [{
            "name": "Accuracy",
            "values": accuracies,
            "color": "#4169E1"
        }]
    }

def generate_html_report(optimization_data: Dict[str, Any], output_path: str) -> str:
    """Generate HTML report with visualizations"""
    
    html_template = """
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gemini Prompt Optimizer - Results Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        h1 {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 30px;
        }
        h2 {
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            font-size: 1.2em;
        }
        .summary-card .value {
            font-size: 2em;
            font-weight: bold;
        }
        .chart-container {
            margin: 30px 0;
            padding: 20px;
            background-color: #fafafa;
            border-radius: 8px;
        }
        .chart-wrapper {
            position: relative;
            height: 400px;
            margin: 20px 0;
        }
        .details-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .details-table th,
        .details-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .details-table th {
            background-color: #3498db;
            color: white;
        }
        .details-table tr:nth-child(even) {
            background-color: #f2f2f2;
        }
        .timestamp {
            text-align: center;
            color: #7f8c8d;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Gemini Prompt Optimizer Results</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Final Accuracy</h3>
                <div class="value">{final_accuracy:.1%}</div>
            </div>
            <div class="summary-card">
                <h3>Best Accuracy</h3>
                <div class="value">{best_accuracy:.1%}</div>
            </div>
            <div class="summary-card">
                <h3>Total Iterations</h3>
                <div class="value">{total_iterations}</div>
            </div>
            <div class="summary-card">
                <h3>Execution Time</h3>
                <div class="value">{execution_time:.1f}s</div>
            </div>
        </div>

        <h2>📈 Optimization Progress</h2>
        <div class="chart-container">
            <div class="chart-wrapper">
                <canvas id="progressChart"></canvas>
            </div>
        </div>

        <h2>🎯 Category Performance</h2>
        <div class="chart-container">
            <div class="chart-wrapper">
                <canvas id="categoryChart"></canvas>
            </div>
        </div>

        <h2>❌ Error Patterns</h2>
        <div class="chart-container">
            <div class="chart-wrapper">
                <canvas id="errorChart"></canvas>
            </div>
        </div>

        <h2>📊 Detailed Results</h2>
        <table class="details-table">
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
                <tr><td>Target Accuracy</td><td>{target_accuracy:.1%}</td></tr>
                <tr><td>Convergence Achieved</td><td>{'✅ Yes' if convergence_achieved else '❌ No'}</td></tr>
                <tr><td>Best Prompt Version</td><td>{best_prompt_version}</td></tr>
                <tr><td>Final Prompt Path</td><td>{final_prompt_path}</td></tr>
            </tbody>
        </table>

        <div class="timestamp">
            Generated on {timestamp}
        </div>
    </div>

    <script>
        // Progress Chart
        const progressCtx = document.getElementById('progressChart').getContext('2d');
        new Chart(progressCtx, {{
            type: 'line',
            data: {{
                labels: {progress_labels},
                datasets: [{{
                    label: 'Accuracy',
                    data: {progress_data},
                    borderColor: '#2E8B57',
                    backgroundColor: 'rgba(46, 139, 87, 0.1)',
                    tension: 0.4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Category Performance Chart
        const categoryCtx = document.getElementById('categoryChart').getContext('2d');
        new Chart(categoryCtx, {{
            type: 'radar',
            data: {{
                labels: {category_labels},
                datasets: [{{
                    label: 'Accuracy',
                    data: {category_data},
                    borderColor: '#4169E1',
                    backgroundColor: 'rgba(65, 105, 225, 0.2)',
                    pointBackgroundColor: '#4169E1'
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    r: {{
                        beginAtZero: true,
                        max: 1,
                        ticks: {{
                            callback: function(value) {{
                                return (value * 100).toFixed(0) + '%';
                            }}
                        }}
                    }}
                }}
            }}
        }});

        // Error Patterns Chart
        const errorCtx = document.getElementById('errorChart').getContext('2d');
        new Chart(errorCtx, {{
            type: 'bar',
            data: {{
                labels: {error_labels},
                datasets: [{{
                    label: 'Frequency',
                    data: {error_data},
                    backgroundColor: '#DC143C',
                    borderColor: '#B22222',
                    borderWidth: 1
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    x: {{
                        ticks: {{
                            maxRotation: 45
                        }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
    
    # Extract data for template
    result = optimization_data.get('result', {})
    iteration_history = optimization_data.get('iteration_history', [])
    category_accuracy = optimization_data.get('category_accuracy', {})
    error_patterns = optimization_data.get('error_patterns', {})
    
    # Prepare chart data
    progress_labels = [h.get('iteration', i) for i, h in enumerate(iteration_history, 1)]
    progress_data = [h.get('accuracy', 0) for h in iteration_history]
    
    category_names = {
        "type": "유형",
        "polarity": "극성", 
        "tense": "시제",
        "certainty": "확실성"
    }
    category_labels = [category_names.get(cat, cat) for cat in category_accuracy.keys()]
    category_data = [category_accuracy[cat].get('accuracy', 0) for cat in category_accuracy.keys()]
    
    # Top 10 error patterns
    sorted_errors = sorted(error_patterns.items(), key=lambda x: x[1], reverse=True)[:10]
    error_labels = [pattern for pattern, count in sorted_errors]
    error_data = [count for pattern, count in sorted_errors]
    
    # Format template
    html_content = html_template.format(
        final_accuracy=result.get('final_accuracy', 0),
        best_accuracy=result.get('best_accuracy', 0),
        total_iterations=result.get('total_iterations', 0),
        execution_time=result.get('execution_time', 0),
        target_accuracy=optimization_data.get('target_accuracy', 0.95),
        convergence_achieved=result.get('convergence_achieved', False),
        best_prompt_version=result.get('best_prompt_version', 0),
        final_prompt_path=result.get('final_prompt_path', 'N/A'),
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        progress_labels=json.dumps(progress_labels),
        progress_data=json.dumps(progress_data),
        category_labels=json.dumps(category_labels),
        category_data=json.dumps(category_data),
        error_labels=json.dumps(error_labels),
        error_data=json.dumps(error_data)
    )
    
    # Write HTML file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"HTML report generated: {output_path}")
    return output_path

def create_summary_dashboard_data(optimization_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create summary dashboard data"""
    result = optimization_data.get('result', {})
    iteration_history = optimization_data.get('iteration_history', [])
    
    dashboard = {
        "summary": {
            "final_accuracy": result.get('final_accuracy', 0),
            "best_accuracy": result.get('best_accuracy', 0),
            "total_iterations": result.get('total_iterations', 0),
            "execution_time": result.get('execution_time', 0),
            "convergence_achieved": result.get('convergence_achieved', False)
        },
        "charts": {
            "progress": create_progress_chart_data(iteration_history),
            "category_performance": create_category_performance_data(
                optimization_data.get('category_accuracy', {})
            ),
            "error_patterns": create_error_pattern_data(
                optimization_data.get('error_patterns', {})
            ),
            "accuracy_distribution": create_accuracy_distribution_data(iteration_history)
        },
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "total_samples": optimization_data.get('total_samples', 0),
            "target_accuracy": optimization_data.get('target_accuracy', 0.95)
        }
    }
    
    return dashboard

def export_visualization_data(optimization_data: Dict[str, Any], output_dir: str) -> Dict[str, str]:
    """Export all visualization data and generate reports"""
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = {}
    
    try:
        # Create dashboard data
        dashboard = create_summary_dashboard_data(optimization_data)
        
        # Export dashboard JSON
        dashboard_path = os.path.join(output_dir, "dashboard.json")
        with open(dashboard_path, 'w', encoding='utf-8') as f:
            json.dump(dashboard, f, ensure_ascii=False, indent=2)
        exported_files['dashboard'] = dashboard_path
        
        # Generate HTML report
        html_path = os.path.join(output_dir, "optimization_report.html")
        generate_html_report(optimization_data, html_path)
        exported_files['html_report'] = html_path
        
        # Export individual chart data
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        for chart_name, chart_data in dashboard["charts"].items():
            if chart_data:  # Only export if data exists
                chart_path = os.path.join(charts_dir, f"{chart_name}.json")
                with open(chart_path, 'w', encoding='utf-8') as f:
                    json.dump(chart_data, f, ensure_ascii=False, indent=2)
                exported_files[f'chart_{chart_name}'] = chart_path
        
        logger.info(f"Visualization data exported to {output_dir}")
        
    except Exception as e:
        logger.error(f"Failed to export visualization data: {e}")
    
    return exported_files