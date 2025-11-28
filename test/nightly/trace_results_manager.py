"""Manager for tracking trace test results across runs."""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional


class TraceResultsManager:
    """Manages success and failure results for trace tests."""

    def __init__(self, results_dir: str = "performance_profiles_neurips_traces"):
        self.results_dir = results_dir
        self.success_file = os.path.join(results_dir, "success_results.json")
        self.failure_file = os.path.join(results_dir, "failure_results.json")

        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Load existing results
        self.success_results = self._load_json(self.success_file)
        self.failure_results = self._load_json(self.failure_file)

    def _load_json(self, filepath: str) -> Dict:
        """Load JSON file or return empty dict if doesn't exist."""
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load {filepath}: {e}")
                return {}
        return {}

    def _save_json(self, filepath: str, data: Dict) -> None:
        """Save data to JSON file with pretty formatting."""
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    def _make_key(self, model: str, config: str) -> str:
        """Create a unique key for model + config combination."""
        return f"{model}::{config}"

    def record_success(
        self,
        model: str,
        config: str,
        batch_size: int,
        input_len: int,
        output_len: int,
        latency_s: float,
        input_throughput: float,
        output_throughput: float,
        itl_ms: Optional[float] = None,
        profile_extend: Optional[str] = None,
        profile_decode: Optional[str] = None,
        extra_metrics: Optional[Dict] = None,
    ) -> None:
        """Record a successful test result.

        Args:
            model: Model name (e.g., "Qwen3-235B")
            config: Configuration (e.g., "TP8_EP2_auto")
            batch_size: Batch size used
            input_len: Input length
            output_len: Output length
            latency_s: Total latency in seconds
            input_throughput: Input throughput (tokens/sec)
            output_throughput: Output throughput (tokens/sec)
            itl_ms: Inter-token latency in milliseconds
            profile_extend: Path to extend profile
            profile_decode: Path to decode profile
            extra_metrics: Any additional metrics to store
        """
        key = self._make_key(model, config)

        result = {
            "model": model,
            "config": config,
            "batch_size": batch_size,
            "input_len": input_len,
            "output_len": output_len,
            "latency_s": round(latency_s, 2),
            "input_throughput_tok_per_s": round(input_throughput, 2),
            "output_throughput_tok_per_s": round(output_throughput, 2),
            "last_updated": datetime.now().isoformat(),
        }

        if itl_ms is not None:
            result["itl_ms"] = round(itl_ms, 2)

        if profile_extend:
            result["profile_extend"] = profile_extend

        if profile_decode:
            result["profile_decode"] = profile_decode

        if extra_metrics:
            result["extra_metrics"] = extra_metrics

        # Update or add new result
        self.success_results[key] = result

        # Remove from failures if it was there
        if key in self.failure_results:
            del self.failure_results[key]

        # Save both files
        self._save_json(self.success_file, self.success_results)
        self._save_json(self.failure_file, self.failure_results)

        print(f"✓ Recorded success: {key}")

    def record_failure(
        self,
        model: str,
        config: str,
        error_message: str,
        error_type: Optional[str] = None,
    ) -> None:
        """Record a failed test result.

        Args:
            model: Model name
            config: Configuration
            error_message: Concise error message
            error_type: Type of error (e.g., "OOM", "Timeout", "ServerError")
        """
        key = self._make_key(model, config)

        # Truncate error message if too long
        if len(error_message) > 500:
            error_message = error_message[:497] + "..."

        result = {
            "model": model,
            "config": config,
            "error_message": error_message,
            "last_updated": datetime.now().isoformat(),
        }

        if error_type:
            result["error_type"] = error_type

        # Update or add new failure
        self.failure_results[key] = result

        # Save
        self._save_json(self.failure_file, self.failure_results)

        print(f"⚠️  Recorded failure: {key} - {error_type or 'Error'}")

    def get_success_count(self) -> int:
        """Get total number of successful configs."""
        return len(self.success_results)

    def get_failure_count(self) -> int:
        """Get total number of failed configs."""
        return len(self.failure_results)

    def get_result(self, model: str, config: str) -> Optional[Dict]:
        """Get result for a specific model+config (success or failure)."""
        key = self._make_key(model, config)
        if key in self.success_results:
            return {"status": "success", **self.success_results[key]}
        elif key in self.failure_results:
            return {"status": "failure", **self.failure_results[key]}
        return None

    def print_summary(self) -> None:
        """Print a summary of all results."""
        print(f"\n{'='*80}")
        print("TRACE RESULTS SUMMARY")
        print(f"{'='*80}")
        print(f"Total successful configs: {self.get_success_count()}")
        print(f"Total failed configs: {self.get_failure_count()}")

        if self.success_results:
            print(f"\n✓ Successful configurations:")
            for key in sorted(self.success_results.keys()):
                result = self.success_results[key]
                print(
                    f"  - {result['model']} ({result['config']}): "
                    f"{result['output_throughput_tok_per_s']:.2f} tok/s"
                )

        if self.failure_results:
            print(f"\n⚠️  Failed configurations:")
            for key in sorted(self.failure_results.keys()):
                result = self.failure_results[key]
                error_type = result.get("error_type", "Error")
                print(f"  - {result['model']} ({result['config']}): {error_type}")

        print(f"{'='*80}\n")

    def generate_markdown_table(self) -> str:
        """Generate a markdown table of all successful results."""
        if not self.success_results:
            return "No successful results yet.\n"

        # Group by model
        by_model: Dict[str, List[Dict]] = {}
        for result in self.success_results.values():
            model = result["model"]
            if model not in by_model:
                by_model[model] = []
            by_model[model].append(result)

        markdown = ""
        for model in sorted(by_model.keys()):
            markdown += f"\n### {model}\n\n"
            markdown += "| Config | Batch Size | Input Len | Latency (s) | Input Throughput (tok/s) | Output Throughput (tok/s) | ITL (ms) |\n"
            markdown += "|--------|-----------|-----------|-------------|--------------------------|---------------------------|----------|\n"

            results = sorted(by_model[model], key=lambda x: x["config"])
            for r in results:
                itl = f"{r['itl_ms']:.2f}" if "itl_ms" in r else "n/a"
                markdown += (
                    f"| {r['config']} | {r['batch_size']} | {r['input_len']} | "
                    f"{r['latency_s']:.2f} | {r['input_throughput_tok_per_s']:.2f} | "
                    f"{r['output_throughput_tok_per_s']:.2f} | {itl} |\n"
                )

        return markdown
