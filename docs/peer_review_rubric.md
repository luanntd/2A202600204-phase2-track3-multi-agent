# Peer Review Rubric

Mỗi nhóm review repo/trace của một nhóm khác trong 8 phút.

| Tiêu chí | Câu hỏi | Điểm |
|---|---|---:|
| Role clarity | Mỗi agent có nhiệm vụ rõ, không overlap quá nhiều không? | 0-2 |
| State design | Shared state có đủ thông tin để handoff mà không mất context không? | 0-2 |
| Failure guard | Có max iterations, timeout, retry/fallback, validation không? | 0-2 |
| Benchmark | Có so sánh single vs multi-agent bằng metric cụ thể không? | 0-2 |
| Trace explanation | Nhóm giải thích được trace: ai làm gì, tốn bao nhiêu, sai ở đâu không? | 0-2 |

## Feedback format

```text
Strength:
Risk / failure mode:
One concrete improvement:
Score:
```
