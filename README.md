# ollama-benchmark



Python Streamlit application for benchmarking Large Language Models (LLMs) using Ollama. 

```bash
streamlit run llm_benchmark_tool.py
```

This is a well-structured tool that:

1. **Detects GPU information** (including Jetson devices)
2. **Loads models from JSON configuration** - models.json-
3. **Filters models based on available VRAM**
4. **Allows users to select which models to benchmark**
5. **Pulls missing models automatically**
6. **Runs benchmarks with real-time progress tracking**
7. **Displays results in tables and charts**
8. **Saves results to CSV**  - benchmark_results.csv -

models.json

```json
{
  "llama3.2:1b": 1.3,
  "llama3.2:3b": 2.0,
  "gemma3:4b": 3.3,
  "gemma3:12b-it-qat":8.9,
  "qwen3:14b": 9.3,
  "gpt-oss:20b": 14,
  "gemma3:27b": 17,
  "qwen3:32b": 20,
  "llama3:70b": 40,
  "gpt-oss:120b": 65
}
```



benchmark_results.csv

| Date             | Model       | Mean Output Speed (tokens/sec) | Mean Prompt Speed (tokens/sec) | GPU                     | GPU Memory (MB) |
| ---------------- | ----------- | ------------------------------ | ------------------------------ | ----------------------- | --------------- |
| 2025-09-02 14:32 | llama3.2:1b | 227.492                        | 17828.728                      | NVIDIA GeForce RTX 3090 | 24576           |

