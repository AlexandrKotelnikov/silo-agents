# SiloAgents benchmark report

Cases: **7**

| Mode | Routing | Task | Leakage | Contamination | Abstention | Provenance | Mean payload chars |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shared_rag` | 50.0% | 100.0% | 85.7% | 42.9% | 100.0% | 100.0% | 822 |
| `isolated_rag` | 100.0% | 100.0% | 85.7% | 0.0% | 100.0% | 100.0% | 783 |
| `policy_gated` | 100.0% | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% | 740 |

Latency is measured for the deterministic local harness only. Token cost is intentionally not reported until a real LLM is connected.
