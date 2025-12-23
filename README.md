# Bài toán chính: Bài toán dịch máy ngôn ngữ VI_EN với Transformer(Code from scratch)
# Giới thiệu
Dự án này triển khai mô hình Transformer từ đầu bằng PyTorch, 
bao gồm đầy đủ các thành phần: Multi-Head Attention, Positional Encoding,
Encoder–Decoder, và quá trình huấn luyện cho bài toán dịch máy. 
# Mục tiêu
- Hiểu rõ kiến trúc Transformer
- Thay đổi tham số và cách tiếp cận để tìm ra phương pháp cải thiện mô hình cũng như BLEU score
- Áp dụng kiến trúc vào bài toán thực tế

# 1. Dữ liệu
- Cặp ngôn ngữ: EN_VI
- Source: 
  + Hugging Face 'PhoMT': https://huggingface.co/datasets/ura-hcmut/PhoMT
  + Train/Val/Test: 3,000,000/20000/20000
# 2. Cấu trúc thư mục
- Gồm 8 model với việc tiếp cận theo các hướng khác nhau để phát triển lên từ baseline
- transformer_baseline.ipynb : Đây là file chứa cấu trúc cơ bản sử dụng cơ chế Post-Layer Normalization sẽ là điểm khởi đầu để phát triển các mô hình tiếp theo.
- model2.ipynb : Mô hình này tối ưu hóa các tham số từ baseline với noam scheduler và labelSmoothing
- model3.ipynb : Kĩ thuật áp dụng : GEGLUFeedForward + AMP + EMA 
- model4.ipynb : Kĩ thuật áp dụng : Transformer enhanced sử dụng Relative positional encoding, LayerDrop, Head dropout
- model5.ipynb : Kĩ thuật áp dụng : Weight tying kết hợp Scheduled sampling và Noam LR & Cosine decay
- model6.ipynb: Kĩ thuật áp dụng : Deep Network
- model7.ibynb: Kĩ thuật áp dụng : Unrestrained Transformer
# 3. Kết quả Bleu và Gemini
  <img width="406" height="561" alt="image" src="https://github.com/user-attachments/assets/cc462422-35c5-4266-8269-afc4f8665779" />

# Bài toán phụ :VLSP Shared Task Machine Translation
# Mục tiêu :
Áp dụng từ bài toán chính xây dựng hệ thống dịch máy chất lượng cao cho lĩnh vực y tế, một lĩnh vực với đặc thù độ phức tạp cao và thuật ngữ chuyên môn.
# Dữ liệu :
Bộ dữ liệu MedicalDataset_VLSP được cung cấp sẽ chia theo tỉ lệ 9/1.
Train/Val/Test : 450,000/50,000/3,000
# Các kĩ thuật tokenize:
- Sentenpiece với thuật toán Unigram:
+ Vocab size: 24000 cho mỗi ngôn ngữ
+ Byte-fallback: Quan trọng trong việc bảo toàn thông tin về tên riêng và các từ chưa từng gặp
+ Character covarage: 99.95% cho tiếng Việt
# Kiến trúc:
+ Pre-Layer Normalization
+ GEGLU FeedForward
+ Weight Typing
# Kết quả
- BLEU: 42.82
- TER: 49.45
- METEOR: 0.64
- Phân tích lỗi cho thấy mô hình dịch tương đối chính xác các thuật ngữ chuyên sâu

# Thành viên nhóm
- Đào Mạnh Phú
- Đoàn Khánh Nhật
- Đặng Đức Minh
	





