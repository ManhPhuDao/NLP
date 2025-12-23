# Bài toán chính: Bài toán dịch máy ngôn ngữ VI_EN với Transformer(Code from scratch)
## Giới thiệu
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
- model1.ipynb : Kĩ thuật áp dụng: Label smoothing + dropout tuning + weight decay + Noam + early stopping
- model2.ipynb : Kĩ thuật áp dụng : AMP + EMA và chỉnh sửa các tham số của model
- model4.ipynb : Kĩ thuật áp dụng : Mạng FFN sử dụng biến thể Gated Linear Unit
- model6.ipynb : Kĩ thuật áp dụng : Transformer enhanced sử dụng Relative positional encoding, LayerDrop, Head dropout
- model8.ipynb: Kĩ thuật áp dụng : Unrestrained Transformer 



