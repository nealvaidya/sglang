"""Trace GLM-4.6 performance with different TP sizes and MoE backends."""

import unittest

from nightly_utils import NightlyBenchmarkRunner
from trace_results_manager import TraceResultsManager

from sglang.test.test_utils import DEFAULT_URL_FOR_TEST

# Shared profile directory for all trace tests
PROFILE_DIR = "performance_profiles_neurips_traces"

MODEL_NAME = "GLM-4.6"

MODEL_CONFIG = {
    "path": "zai-org/GLM-4.6",
    "extra_args": ["--trust-remote-code"],
}

BATCH_SIZES = [1]
INPUT_LENS = (4096,)
OUTPUT_LENS = (512,)

# Test configurations: (tp_size, ep_size, moe_backends)
# GLM 4.6 is large (357B), so only TP8
CONFIGS = [
    (8, 2, ["auto", "triton"]),  # TP8, EP2
]


class TraceGLM46(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.runner = NightlyBenchmarkRunner(
            PROFILE_DIR, "GLM-4.6 Performance Traces", cls.base_url
        )
        cls.runner.setup_profile_directory()
        cls.results_manager = TraceResultsManager(PROFILE_DIR)

    def test_glm_46_configs(self):
        """Test GLM-4.6 with different TP/EP/MoE configurations."""

        all_results = []
        successful_configs = []
        failed_configs = []

        for tp_size, ep_size, moe_backends in CONFIGS:
            for moe_backend in moe_backends:
                server_args = ["--tp", str(tp_size)] + MODEL_CONFIG["extra_args"]

                if ep_size > 1:
                    server_args += ["--ep", str(ep_size)]

                server_args += ["--moe-runner-backend", moe_backend]

                ep_str = f"_EP{ep_size}" if ep_size > 1 else ""
                variant = f"TP{tp_size}{ep_str}_{moe_backend}"
                config_name = f"glm-4.6 {variant}"

                print(f"\n{'='*80}")
                print(f"Running {config_name}...")
                print(f"{'='*80}\n")

                try:
                    results, success = self.runner.run_benchmark_for_model(
                        model_path=MODEL_CONFIG["path"],
                        batch_sizes=BATCH_SIZES,
                        input_lens=INPUT_LENS,
                        output_lens=OUTPUT_LENS,
                        other_args=server_args,
                        variant=variant,
                        extra_bench_args=["--trust-remote-code"],
                    )

                    if success and results:
                        all_results.extend(results)
                        self.runner.add_report(results)
                        successful_configs.append(config_name)
                        print(f"✓ Success: {config_name}")

                        # Record success in results manager
                        for result in results:
                            self.results_manager.record_success(
                                model=MODEL_NAME,
                                config=variant,
                                batch_size=result.batch_size,
                                input_len=result.input_len,
                                output_len=result.output_len,
                                latency_s=result.total_latency,
                                input_throughput=result.prefill_throughput,
                                output_throughput=result.decode_throughput,
                                itl_ms=(
                                    result.inter_token_latency * 1000
                                    if result.inter_token_latency
                                    else None
                                ),
                            )
                    else:
                        failed_configs.append(config_name)
                        print(f"⚠️  Failed: {config_name}")
                        # Record failure
                        self.results_manager.record_failure(
                            model=MODEL_NAME,
                            config=variant,
                            error_message="Benchmark failed to complete",
                            error_type="BenchmarkFailure",
                        )
                except Exception as e:
                    failed_configs.append(config_name)

                    error_msg = str(e)
                    print(f"⚠️  Error: {error_msg}")

                    # Determine error type
                    error_type = "Unknown"
                    if (
                        "out of memory" in error_msg.lower()
                        or "oom" in error_msg.lower()
                    ):
                        error_type = "OOM"
                    elif "timeout" in error_msg.lower():
                        error_type = "Timeout"
                    elif "server failed" in error_msg.lower():
                        error_type = "ServerError"

                    # Record failure
                    self.results_manager.record_failure(
                        model=MODEL_NAME,
                        config=variant,
                        error_message=error_msg,
                        error_type=error_type,
                    )

        # Write final report
        self.runner.write_final_report()

        # Print results manager summary
        self.results_manager.print_summary()

        print(f"\n{'='*80}")
        print(f"THIS RUN SUMMARY")
        print(f"{'='*80}")
        print(f"Successful: {len(successful_configs)}")
        print(f"Failed: {len(failed_configs)}")

        if successful_configs:
            print(f"\n✓ Successful:")
            for config in successful_configs:
                print(f"  - {config}")

        # Print markdown table
        print(f"\n{self.results_manager.generate_markdown_table()}")

        if failed_configs:
            print(f"\n⚠️  Failed:")
            for config in failed_configs:
                print(f"  - {config}")


if __name__ == "__main__":
    unittest.main()
