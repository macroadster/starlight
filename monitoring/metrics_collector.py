import json
from datetime import datetime

class MetricsCollector:
    def __init__(self, log_file='monitoring/metrics.jsonl'):
        self.log_file = log_file

    def log_metrics(self, metrics):
        """Log metrics as a JSON line"""
        entry = {
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')

    def get_recent_metrics(self, limit=10):
        """Get recent metrics entries"""
        try:
            with open(self.log_file, 'r') as f:
                lines = f.readlines()[-limit:]
            return [json.loads(line.strip()) for line in lines]
        except FileNotFoundError:
            return []