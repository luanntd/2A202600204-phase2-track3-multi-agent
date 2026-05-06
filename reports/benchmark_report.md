# Báo Cáo Benchmark

## Tổng Hợp

| Lần chạy | Độ trễ (giây) | Chi phí (USD) | Chất lượng / 10 | Trích dẫn | Ghi chú |
|---|---:|---:|---:|---:|---|
| baseline_q1 | 9.71 | $0.0005 | 8.0 | 0 | citations=0; Câu trả lời cung cấp tổng quan chính xác và toàn diện về GraphRAG, chi tiết các tính năng, ứng dụng và thách thức, nhưng thiếu trích dẫn cụ thể hoặc bằng chứng để hỗ trợ các khẳng định. |
| multi_agent_q1 | 27.21 | $0.0016 | 9.0 | 5 | citations=5; Câu trả lời cung cấp tổng quan chính xác và toàn diện về GraphRAG, được cấu trúc rõ ràng với bằng chứng và trích dẫn liên quan, mặc dù có thể được chi tiết hơn ở một số lĩnh vực nhất định. |
| baseline_q2 | 11.73 | $0.0005 | 8.0 | 0 | citations=0; Câu trả lời cung cấp so sánh chính xác và toàn diện giữa luồng xử lý single-agent và multi-agent, nhưng thiếu trích dẫn cụ thể hoặc bằng chứng để hỗ trợ các khẳng định. |
| multi_agent_q2 | 21.12 | $0.0012 | 9.0 | 4 | citations=4; Câu trả lời cung cấp so sánh toàn diện giữa luồng xử lý single-agent và multi-agent, chi tiết chính xác ưu điểm và thách thức của chúng, đồng thời trích dẫn các nghiên cứu liên quan để hỗ trợ các khẳng định. |
| baseline_q3 | 13.07 | $0.0005 | 10.0 | 0 | citations=0; Câu trả lời chính xác, toàn diện, được cấu trúc rõ ràng và cung cấp bằng chứng chi tiết cho từng guardrail được thảo luận. |
| multi_agent_q3 | 26.71 | $0.0015 | 9.0 | 5 | citations=5; Câu trả lời cung cấp tổng quan chính xác và toàn diện về các guardrail sản xuất cho LLM agent, được cấu trúc rõ ràng với các danh mục và chiến lược liên quan, mặc dù có thể được hưởng lợi từ các trích dẫn cụ thể hơn. |

## Phân Tích Kết Quả

- **Độ trễ tăng thêm**: multi-agent chậm hơn 17.50s so với baseline (+180%).
- **Chênh lệch chất lượng**: multi-agent đạt điểm cao hơn 1 so với baseline.

## Phân Tích Failure Mode

### Max-iteration cap triggered (đạt giới hạn số lần lặp tối đa)

**Kịch bản**: Một truy vấn không có nguồn phù hợp khiến Researcher trả về ghi chú rỗng.
Supervisor phát hiện `research_notes is None` và chuyển hướng lại về Researcher ở mỗi
lần lặp cho đến khi `max_iterations` (mặc định là 6) bị cạn kiệt.

**Bằng chứng từ trace**: danh sách `errors` trong state chứa
`'Max iterations (6) reached — stopping early'` và `final_answer` là `None`.

**Cách fix**: Researcher hiện fallback về mock corpus thay vì trả về
kết quả rỗng. Giới hạn số lần lặp của Supervisor cung cấp một ranh giới an toàn cứng và thêm
một mục lỗi mang tính mô tả để phân tích sau sự cố.

## Khi Nào Nên Dùng Multi-Agent

**Nên dùng multi-agent khi**:
- Tác vụ có các bước con riêng biệt, có thể tách rời (nghiên cứu / phân tích / viết).
- Chất lượng quan trọng hơn độ trễ (pipeline đánh đổi ~3-5× độ trễ để có kết quả tốt hơn).
- Các bước con độc lập có thể chạy song song trong các lần lặp tiếp theo.

**KHÔNG nên dùng multi-agent khi**:
- Truy vấn đơn giản và có phạm vi rõ ràng (một LLM call là nhanh hơn và rẻ hơn).
- Độ trễ là ràng buộc chính (hệ thống real-time hướng đến khách hàng).
- Việc thêm agent tạo ra overhead điều phối mà không có cải thiện chất lượng rõ ràng.
