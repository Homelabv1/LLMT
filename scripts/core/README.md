# Core Modules

This package provides the foundational components for the LLM Testing Framework.

## Modules

### questions.py

Handles loading and validating test questions.

**Public API**:
- `Question` - Question data class
- `load_questions(filepath)` - Load questions from JSON
- `validate_questions(questions)` - Validate question structure
- `get_questions_by_category(questions, category)` - Filter by category
- `load_inventory(filepath)` - Load inventory JSON
- `create_system_prompt(inventory)` - Create system prompt with inventory

### results.py

Handles saving and loading test results with atomic file operations.

**Public API**:
- `QuestionResult` - Per-question result data class
- `RunMetadata` - Test run metadata
- `ResultsWriter` - Thread-safe JSONL writer
- `load_results(filepath)` - Load results from JSONL
- `get_completed_question_ids(results_dir, model)` - Get completed IDs
- `append_failure_log(...)` - Log failures to CSV
- `write_load_failure(...)` - Write load failure JSON

### metrics.py

GPU monitoring and cooldown logic using nvidia-smi.

**Public API**:
- `GPUMetrics` - GPU metrics snapshot
- `GPUMonitor` - Background metrics collector
- `get_gpu_info(gpu_index)` - Get GPU details
- `get_gpu_metrics(gpu_index)` - Get current metrics
- `wait_for_cooldown(...)` - Dynamic cooldown
- `apply_power_limits(limits)` - Set power limits
- `PowerMonitor` - Background power monitoring

### runner.py

Test execution engine with signal handling and resumption.

**Public API**:
- `TestConfig` - Test configuration
- `TestRunner` - Main test runner
- `run_single_test(...)` - Convenience function
- `load_models_from_config(filepath)` - Load model list
- `print_batch_status(...)` - Print status table

## Data Flow

```
                    ┌──────────────┐
                    │ TestRunner   │
                    └──────┬───────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
        ▼                  ▼                  ▼
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│  questions.py │  │   results.py  │  │   metrics.py  │
│               │  │               │  │               │
│ Load questions│  │ Write JSONL   │  │ GPU monitoring│
│ Load inventory│  │ Track status  │  │ Cooldown      │
└───────────────┘  └───────────────┘  └───────────────┘
```

## Extension Points

### Adding New Question Categories

1. Update `validate_questions()` in questions.py
2. Add questions to `homelab_test_questions.json`

### Custom System Prompt

```python
from core.questions import create_system_prompt

custom_template = """You are an assistant...
INVENTORY: {inventory_json}
"""
prompt = create_system_prompt(inventory, template=custom_template)
```

### Custom Metrics

Subclass `GPUMonitor` to add custom metric collection:

```python
class CustomMonitor(GPUMonitor):
    def _monitor_loop(self):
        # Custom monitoring logic
        pass
```
