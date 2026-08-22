## CPU latency (median of 20, ms)

platform arm64 / Darwin, torch 2.8.0 (8 threads), onnxruntime 1.25.1; torch.compile not measured (--skip-compile: inductor's C++ backend stalled on this macOS CPU setup)

| refiner encoder | torch eager | torch.compile | ORT fp32 | ORT weight-only int8 | ORT dynamic int8 |
|---|---|---|---|---|---|
| n = 5,000 | 22.6 | nan | 55.5 | 56.1 | 53.8 |
| n = 20,000 | 86.2 | nan | 233.3 | 235.0 | 224.3 |

| GPT-2 124M, batch 1, seq 128 | torch eager | torch.compile | ORT fp32 | ORT weight-only int8 | ORT dynamic int8 |
|---|---|---|---|---|---|
| one forward | 65.5 | nan | 78.5 | 51.7 | 30.9 |
