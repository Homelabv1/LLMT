# Test Data

This directory contains the test data for the LLM Testing Framework.

## Files

### homelab_inventory.json

The homelab inventory dataset containing:

| Category | Count | Description |
|----------|-------|-------------|
| Servers | 10 | Production servers (Proxmox, NAS, K8s, etc.) |
| Desktops | 5 | Workstations and PCs |
| Mini PCs | 5 | Low-power systems |
| LLM Rigs | 5 | GPU-equipped testing systems |
| Switches | 2 | Network switches |
| Spare Parts | Various | RAM, storage, GPUs, etc. |

### homelab_test_questions.json

42 test questions across 8 categories:

| Category | Count | Description |
|----------|-------|-------------|
| simple_lookup | 8 | Direct retrieval (IP, RAM, CPU) |
| aggregation | 6 | Sum/count operations |
| filtering | 6 | Conditional logic |
| compatibility | 6 | Hardware relationships |
| complex_multistep | 4 | Chained reasoning |
| natural_language | 5 | Informal queries |
| inventory_management | 3 | Change scenarios |
| error_handling | 4 | Missing/invalid data |

## Schema

### Question Format

```json
{
  "q_id": 1,
  "category": "simple_lookup",
  "question": "What is the IP address of pve-main?",
  "expected": "192.168.1.10",
  "difficulty": "easy"
}
```

### Difficulty Levels

- `easy`: Direct lookups, single fact retrieval
- `medium`: Aggregations, filtering, basic reasoning
- `hard`: Multi-step reasoning, complex comparisons

## Customization

### Adding Questions

Add to the `questions` array in `homelab_test_questions.json`:

```json
{
  "q_id": 43,
  "category": "simple_lookup",
  "question": "Your question here",
  "expected": "Expected answer",
  "difficulty": "easy"
}
```

### Adding Inventory Items

Add to the appropriate category in `homelab_inventory.json`:

```json
{
  "name": "new-server-01",
  "type": "server",
  "ip": "192.168.1.100",
  "cpu": {"model": "...", "cores": 8, "threads": 16},
  "ram_gb": 64,
  ...
}
```

### Custom Datasets

Create your own files following the same schema:

```python
python scripts/run_test.py \
    --questions /path/to/my_questions.json \
    --inventory /path/to/my_inventory.json
```

## Validation

Questions are validated on load:
- Unique `q_id` values
- Valid `category` from allowed list
- Non-empty `question` and `expected`
- Valid `difficulty` (easy/medium/hard)
