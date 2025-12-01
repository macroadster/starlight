"""
Starlight Real-Time Dashboard Generator
Generates and updates performance dashboards from metrics data.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any


class DashboardGenerator:
    """Generates real-time performance dashboards"""

    def __init__(self, dashboard_path: str = 'docs/grok/performance_dashboard.md'):
        self.dashboard_path = Path(dashboard_path)

    def update_dashboard(self, metrics: Dict[str, Any]) -> bool:
        """Update the performance dashboard markdown with new metrics"""
        if not self.dashboard_path.exists():
            print(f"Dashboard file not found: {self.dashboard_path}")
            return False

        # Read current content
        with open(self.dashboard_path, 'r') as f:
            content = f.read()

        # Update placeholders
        content = content.replace(
            '**Last Updated**: 2025-11-22 14:32:15 UTC',
            f'**Last Updated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC'
        )
        content = content.replace('**Model**: v4_production_quantized', '**Model**: v4_production_quantized')  # keep as is
        content = content.replace('**Status**: ✅ HEALTHY', '**Status**: ✅ HEALTHY')  # keep as is

        # Update FPR
        fpr_line = f"| False Positive Rate | {metrics['fpr']:.2f}% | <0.05% | {'⚠️ Close' if metrics['fpr'] > 0.05 else '✅ Good'} |"
        content = content.replace('| False Positive Rate | 0.07% | <0.05% | ⚠️ Close |', fpr_line)

        # Update detection rates
        methods = ['lsb', 'alpha', 'palette', 'exif', 'eoi']
        for method in methods:
            rate = metrics['detection_rates'].get(method, 0)
            status = '✅ Good' if rate > 99 else '⚠️ Watch'
            line = f"| {method.capitalize()} Detection | {rate:.1f}% | >99% | {status} |"
            old_line = f"| {method.capitalize()} Detection | 98.5% | >99% | ✅ Good |"  # approximate
            content = content.replace(old_line, line)

        # Update performance
        perf = metrics['performance']
        content = content.replace('- **Latency p50**: 3.8ms', f'- **Latency p50**: {perf["p50_ms"]:.1f}ms')
        content = content.replace('- **Latency p95**: 4.2ms (Target: <5ms)', f'- **Latency p95**: {perf["p95_ms"]:.1f}ms (Target: <5ms)')
        content = content.replace('- **Throughput**: 238 img/sec', f'- **Throughput**: {perf["throughput_imgs_per_sec"]:.0f} img/sec')

        # Update dataset quality
        dq = metrics['dataset_quality']
        balance_str = ' | '.join([f"{k} {v:.0f}%" for k, v in dq['balance'].items()])
        content = content.replace('- **Total Samples**: 5,200', f'- **Total Samples**: {dq["total"]}')
        content = content.replace('- **Method Balance**: LSB 22% | Alpha 19% | Palette 18% | EXIF 20% | EOI 21%', f'- **Method Balance**: {balance_str}')
        content = content.replace('- **Shannon Entropy**: 2.47 / 2.32 (excellent diversity)', f'- **Shannon Entropy**: {dq["entropy"]:.2f} / 2.32 (excellent diversity)')

        # Write back
        with open(self.dashboard_path, 'w') as f:
            f.write(content)

        print(f"✅ Dashboard updated: {self.dashboard_path}")
        return True

    def generate_html_dashboard(self, metrics: Dict[str, Any], output_path: str = 'monitoring/dashboard.html') -> bool:
        """Generate an HTML dashboard for web viewing"""
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Starlight Performance Dashboard</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ background: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .good {{ color: green; }}
        .warning {{ color: orange; }}
        .error {{ color: red; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
    </style>
</head>
<body>
    <h1>Starlight Performance Dashboard</h1>
    <p><strong>Last Updated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC</p>
    <p><strong>Model:</strong> v4_production_quantized</p>
    <p><strong>Status:</strong> ✅ HEALTHY</p>

    <div class="metric">
        <h2>Key Metrics</h2>
        <table>
            <tr><th>Metric</th><th>Value</th><th>Target</th><th>Status</th></tr>
            <tr>
                <td>False Positive Rate</td>
                <td>{metrics['fpr']:.2f}%</td>
                <td>&lt;0.05%</td>
                <td class="{'warning' if metrics['fpr'] > 0.05 else 'good'}">{'⚠️ Close' if metrics['fpr'] > 0.05 else '✅ Good'}</td>
            </tr>
        </table>
    </div>

    <div class="metric">
        <h2>Detection Rates by Method</h2>
        <table>
            <tr><th>Method</th><th>Detection Rate</th><th>Target</th><th>Status</th></tr>
            {"".join([f"<tr><td>{method.capitalize()}</td><td>{metrics['detection_rates'].get(method, 0):.1f}%</td><td>&gt;99%</td><td class='{'good' if metrics['detection_rates'].get(method, 0) > 99 else 'warning'}'>{'✅ Good' if metrics['detection_rates'].get(method, 0) > 99 else '⚠️ Watch'}</td></tr>" for method in ['lsb', 'alpha', 'palette', 'exif', 'eoi']])}
        </table>
    </div>

    <div class="metric">
        <h2>Performance Metrics</h2>
        <ul>
            <li><strong>Latency p50:</strong> {metrics['performance']['p50_ms']:.1f}ms</li>
            <li><strong>Latency p95:</strong> {metrics['performance']['p95_ms']:.1f}ms (Target: &lt;5ms)</li>
            <li><strong>Throughput:</strong> {metrics['performance']['throughput_imgs_per_sec']:.0f} img/sec</li>
        </ul>
    </div>

    <div class="metric">
        <h2>Dataset Quality</h2>
        <ul>
            <li><strong>Total Samples:</strong> {metrics['dataset_quality']['total']}</li>
            <li><strong>Method Balance:</strong> {" | ".join([f"{k} {v:.0f}%" for k, v in metrics['dataset_quality']['balance'].items()])}</li>
            <li><strong>Shannon Entropy:</strong> {metrics['dataset_quality']['entropy']:.2f} / 2.32 (excellent diversity)</li>
        </ul>
    </div>
</body>
</html>
        """

        output_path = Path(output_path)
        output_path.parent.mkdir(exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"✅ HTML dashboard generated: {output_path}")
        return True


def update_dashboard_markdown(metrics: Dict[str, Any]) -> bool:
    """Legacy function for backward compatibility"""
    generator = DashboardGenerator()
    return generator.update_dashboard(metrics)


if __name__ == '__main__':
    # Example usage
    sample_metrics = {
        'fpr': 0.02,
        'detection_rates': {'lsb': 99.5, 'alpha': 98.8, 'palette': 99.2, 'exif': 99.9, 'eoi': 99.1},
        'performance': {'p50_ms': 3.2, 'p95_ms': 4.1, 'throughput_imgs_per_sec': 245},
        'dataset_quality': {'balance': {'lsb': 22, 'alpha': 19, 'palette': 18, 'exif': 20, 'eoi': 21}, 'entropy': 2.47, 'total': 5200}
    }

    generator = DashboardGenerator()
    generator.update_dashboard(sample_metrics)
    generator.generate_html_dashboard(sample_metrics)