# Test Data

This directory contains sample test data for the LLM Testing Framework.

## Files

### homelab_inventory.json

A sample dataset representing a homelab infrastructure. This serves as example context data that gets injected into the system prompt. You can replace this with your own dataset.

| Category | Count | Description |
|----------|-------|-------------|
| Servers | 10 | Production servers |
| Desktops | 5 | Workstations and PCs |
| Mini PCs | 5 | Low-power systems |
| LLM Rigs | 5 | GPU-equipped testing systems |
| Switches | 2 | Network switches |
| Spare Parts | Various | RAM, storage, GPUs, etc. |

### homelab_test_questions.json

42 sample test questions across 8 categories:

| Category | Count | Description |
|----------|-------|-------------|
| simple_lookup | 8 | Direct information retrieval |
| aggregation | 6 | Sum/count operations |
| filtering | 6 | Conditional logic |
| compatibility | 6 | Relationship reasoning |
| complex_multistep | 4 | Chained reasoning |
| natural_language | 5 | Informal query variations |
| inventory_management | 3 | Change scenarios |
| error_handling | 4 | Missing/invalid data |

## Schema

### Question Format

```json
{
  "q_id": 1,
  "category": "simple_lookup",
  "question": "Your question text here",
  "expected": "Expected answer",
  "difficulty": "easy"
}
```

### Difficulty Levels

- `easy`: Direct lookups, single fact retrieval
- `medium`: Aggregations, filtering, basic reasoning
- `hard`: Multi-step reasoning, complex comparisons

## Custom Datasets

The sample data demonstrates the framework's capabilities, but you can use your own:

### Using Custom Questions and Data

```bash
python scripts/run_test.py \
    --questions /path/to/my_questions.json \
    --inventory /path/to/my_data.json
```

### Creating Custom Questions

Create a JSON file with this structure:

```json
{
  "questions": [
    {
      "q_id": 1,
      "category": "simple_lookup",
      "question": "Your question here",
      "expected": "Expected answer",
      "difficulty": "easy"
    }
  ]
}
```

### Creating Custom Context Data

The inventory/context file can be any JSON structure. It gets serialized and injected into the system prompt template in `scripts/core/questions.py`.

## Validation

Questions are validated on load:
- Unique `q_id` values
- Valid `category` from allowed list
- Non-empty `question` and `expected`
- Valid `difficulty` (easy/medium/hard)
