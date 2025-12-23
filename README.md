# I. Bài toán chính: Bài toán dịch máy ngôn ngữ VI_EN với Transformer(Code from scratch)
# 1. Giới thiệu
Dự án này triển khai mô hình Transformer từ đầu bằng PyTorch, 
bao gồm đầy đủ các thành phần: Multi-Head Attention, Positional Encoding,
Encoder–Decoder, và quá trình huấn luyện cho bài toán dịch máy. 
# 2. Mục tiêu
- Hiểu rõ kiến trúc Transformer
- Thay đổi tham số và cách tiếp cận để tìm ra phương pháp cải thiện mô hình cũng như BLEU score
- Áp dụng kiến trúc vào bài toán thực tế

# 3. Dữ liệu
- Cặp ngôn ngữ: EN_VI
- Source: 
  + Hugging Face 'PhoMT': https://drive.google.com/drive/u/1/folders/1PbUlXXUoUTHuhSSxJLsW1nod-PZJyHsR
  + Train/Val/Test: 3,000,000/20000/20000
# 4. Cấu trúc thư mục
- Gồm 8 model với việc tiếp cận theo các hướng khác nhau để phát triển lên từ baseline
- transformer_baseline.ipynb : Đây là file chứa cấu trúc cơ bản sử dụng cơ chế Post-Layer Normalization sẽ là điểm khởi đầu để phát triển các mô hình tiếp theo.
- model2.ipynb : Mô hình này tối ưu hóa các tham số từ baseline với noam scheduler và labelSmoothing
- model3.ipynb : Kĩ thuật áp dụng : GEGLUFeedForward + AMP + EMA 
- model4.ipynb : Kĩ thuật áp dụng : Transformer enhanced sử dụng Relative positional encoding, LayerDrop, Head dropout
- model5.ipynb : Kĩ thuật áp dụng : Weight tying kết hợp Scheduled sampling và Noam LR & Cosine decay
- model6.ipynb: Kĩ thuật áp dụng : Deep Network
- model7.ibynb: Kĩ thuật áp dụng : Unrestrained Transformer
# 5. Kết quả Bleu và Gemini
  | Model    | BleuScore | Gemini Score |
|----------|-----------|--------------|
| Baseline | 9.11      | 35.95        |
| model2   | 12.21     | 47.25        |
| model3   | 19.16     | 62.68        |
| model4   | 15.55     | 36.23        |
| model5   | 16.39     | 46.85        |
| model6   | 25.61     | 56.54        |
| model7   | 36.23     | 59.40        |


# II. Bài toán phụ :VLSP Shared Task Machine Translation
 ## A. Không sử dụng llm model.
# 1. Mục tiêu :
Áp dụng từ bài toán chính xây dựng hệ thống dịch máy chất lượng cao cho lĩnh vực y tế, một lĩnh vực với đặc thù độ phức tạp cao và thuật ngữ chuyên môn.
# 2. Dữ liệu :
Bộ dữ liệu MedicalDataset_VLSP được cung cấp sẽ chia theo tỉ lệ 9/1.
Train/Val/Test : 450,000/50,000/3,000
# 3. Các kĩ thuật tokenize:
- Sentenpiece với thuật toán Unigram:
+ Vocab size: 24000 cho mỗi ngôn ngữ
+ Byte-fallback: Quan trọng trong việc bảo toàn thông tin về tên riêng và các từ chưa từng gặp
+ Character covarage: 99.95% cho tiếng Việt
# 4. Kiến trúc:
+ Pre-Layer Normalization
+ GEGLU FeedForward
+ Weight Typing
# 5. Kết quả
- BLEU: 42.82
- TER: 49.45
- METEOR: 0.64
- Phân tích lỗi cho thấy mô hình dịch tương đối chính xác các thuật ngữ chuyên sâu
## B. Sử dụng llm model.
# 1. Mô hình cơ sở.
- NLLB-200 – là mô hình được huấn luyện chuyên biệt cho việc dịch đa ngữ, phiên bản 600M tham số nhẹ tuy nhiên vẫn đảm bảo được chất lượng.
# 2. Kĩ thuật tinh chỉnh: LoRA
- Cơ chế: Đóng băng mô hình gốc, chỉ thêm vào các ma trận hạng thấp vào các lớp Attention (q_proj, v_proj, k_proj) và Feed-Forward (fc1, fc2).	
- Tham số huấn luyện: Chỉ train khoảng 1-3% tổng số tham số mô hình.
- Thư viện: Sử dụng PEFT của Hugging Face.
# 3. Kết quả
- BLEU Score: 33.6
- METEOR: 0.6199
- TER: 62.31

# Thành viên nhóm
- Đào Mạnh Phú
- Đoàn Khánh Nhật
- Đặng Đức Minh
	





