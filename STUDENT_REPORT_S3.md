# Student Report — S3: Pilot and Switching Accounting

## 1. Phạm vi triển khai

Đối chiếu PDF người dùng cung cấp: `Hierarchical_Multi_Timescale_Deep_Reinforcement_Learning_for_Overhead__and_Switching_Aware_Tri_Hybrid_Beamforming (1).pdf`, trang 11, mục S3. Phần triển khai gồm pilot_cost_bb, pilot_cost_rf, pilot_cost_em, switch_cost_em, switch_cost_rf, net_rate và utility theo (27)–(31), cùng kiểm tra thời điểm hành động RF/EM.

Bản hiện tại gộp phần chạy vào `s3.py`. `main()` chỉ thực thi khi chạy trực tiếp bằng `python s3.py`; import module không tự chạy mô phỏng. Các tham số nằm trong `config.py`: `S3Config` cho accounting, `S3_INTEGRATION` cho lần chạy. Không dùng JSON làm cấu hình chạy nữa.

S1/S2 cung cấp mô hình kênh và precoder; S3 không thêm mô hình kênh hoặc thuật toán tối ưu hành động. Việc gộp file và cập nhật tài liệu không thay đổi các công thức accounting.

## 2. Nguồn dữ liệu và đầu vào tự đặt

- S2: `load_or_generate_trajectory()` tạo hoặc đọc lại quỹ đạo; lấy `alpha[t]` và `aods[t]` theo từng slot.
- S1: `run_fixed_configuration()` nhận kênh, q và beam đang áp dụng để tính gross SE từ kênh/precoder; mọi slot được đánh giá, không lấy mẫu thưa.
- S3: nhận gross SE cùng hành động để tính overhead, switching, net SE và utility.
- Lần chạy mặc định dùng 200 slot đầu của quỹ đạo 16.000 slot, seed 2299, tốc độ mọi user là 30 km/h. Seed 2299 là lựa chọn debug dùng lại trường hợp cache của S2, không phải seed bắt buộc theo PDF hay kết quả test cuối cùng.
- Lịch hành động tại 0, 40, 80, 160 là dữ liệu kiểm thử do người triển khai đặt. Mốc 40 refresh giữ beam; mốc 80 đổi hai beam; mốc 160 đổi EM kèm refresh RF.
- q=5 và beam [34,22,40,35] ban đầu trùng một output chẩn đoán S2 với seed 2301. Trong lần chạy seed 2299, chúng chỉ là cấu hình thử; không được tuyên bố là cấu hình tối ưu của quỹ đạo này.
- Không dùng SE giả lập `20 + (t % 7)` trong đường chạy chính. Unit test riêng vẫn dùng đầu vào tự đặt để kiểm tra từng nhánh.

## 3. Quy ước triển khai

- Pilot BB được tính mỗi slot. Khởi tạo miễn phí RF/EM và switching, kể cả phần phân bổ RF/EM trong kỳ RF đầu tiên; không miễn pilot BB định kỳ. Đây là cách diễn giải “initialization is not charged” đã sử dụng trong implementation.
- Pilot RF và EM được phân bổ trên kỳ T_RF; không phân bổ phí EM trên T_EM.
- Switching chỉ bị tính tại sự kiện thay đổi; utility âm được giữ nguyên.
- Vị trí trong danh sách beam là thứ tự RF chain. Đầu nối với S1 kiểm tra q và beam theo quy ước bắt đầu từ 1; lõi S3 chỉ cần các mã không âm để so sánh trạng thái.
- Đổi EM phải kèm refresh RF, đổi beam phải kèm refresh RF và hành động phải đúng mốc cho phép.
- Giữa các sự kiện giữ cấu hình; việc duy trì phí pilot trong kỳ không có nghĩa là refresh mới ở mọi slot.

## 4. Các kiểm tra theo yêu cầu S3

`check_pdf_requirements()` chạy trên dữ liệu mô phỏng vừa tính, không đọc PDF khi thực thi. Các thử nghiệm thay đổi tau tái sử dụng cùng gross SE, cùng kênh và cùng chuỗi cấu hình.

| Nhóm kiểm tra | Kết quả lần chạy xác minh |
|---|---|
| Ba tau bằng 0 | Sai lệch lớn nhất net SE − gross SE: 0.0; đạt |
| tau RF tăng 32 → 64 | Thay đổi net SE nhỏ nhất −1.1832385527, lớn nhất 0.0; đạt |
| tau EM tăng 64 → 128 | Thay đổi net SE nhỏ nhất −2.2084895858, lớn nhất 0.0; đạt |
| Tỷ lệ pilot mặc định | 0.0714285714 / 0.1285714286 / 0.2428571429; đạt |
| Refresh giữ beam | Tại slot 40 và 160, switching RF bằng 0; đạt |
| Đổi EM cần refresh RF | Slot 160 có refresh; bỏ refresh bị từ chối; đạt |
| Khởi tạo | 40 slot đầu không có phí RF/EM và switching, pilot BB = 1/14; đạt |

Hệ số tăng tau=2 và dung sai 1e-10 là lựa chọn kiểm thử trong config. Cần có đủ tình huống hành động để kiểm tra, không coi tập tình huống rỗng là đạt. Kiểm tra tỷ lệ mặc định hiện yêu cầu T_RF, T_EM và NRF mặc định. Tham số hoặc lịch không tương thích có thể gây lỗi trước khi xuất báo cáo.

Hai file test được giữ: `test_s3.py` có 14 test, `test_s3_integration.py` có 8 test. Kiểm tra bổ sung gồm input sai, reset/thứ tự slot, bảo toàn tài nguyên, SE khớp lời gọi S1 trực tiếp, dữ liệu quỹ đạo không bị sửa, phát hiện thiếu tình huống và sai phí switching. Lần xác minh sau khi gộp file: 22/22 test đạt.

## 5. Kết quả chạy ghép

Lệnh đã dùng:

```bash
python s3.py
python -m unittest test_s3 test_s3_integration
```

| Chỉ số | Giá trị |
|---|---:|
| Số slot đánh giá | 200 |
| Mean gross SE | 15.8277474589 |
| Mean net SE | 13.8017585286 |
| Mean utility | 13.2517585286 |
| Mean pilot fraction | 0.1285714286 |
| RF refresh ngoài khởi tạo | 3 |
| EM switches | 1 |
| Tổng số RF chain thay beam | 2 |
| Nhóm kiểm tra trong chương trình đạt | 7/7 |

Gross/net SE có đơn vị bit/s/Hz. Utility là giá trị sau khi trừ các hình phạt switching. Đây là số liệu của kịch bản ghép thử đặt trước; không đại diện kết quả benchmark/DRL hoặc hiệu năng tối ưu.

## 6. File kết quả và tái lập

`outputs/s3_integration/output.txt` chứa số liệu tổng hợp và kết quả kiểm tra dạng nhóm. Dữ liệu quỹ đạo được cache trong `data/trajectories/` theo mặc định. File output TXT S1/S2 không được dùng làm đầu vào S3.

Hai file test không phải dependency của lệnh `python s3.py` nhưng được giữ để kiểm tra hồi quy. Nếu xóa chúng thì không còn chạy được lệnh unittest ở trên; số 22/22 chỉ mô tả lần xác minh đã thực hiện, không phải số kiểm tra mà s3.py tự chạy.

## 7. Giới hạn và vấn đề cần lưu ý

- Chưa chạy baseline/DRL, nhiều seed hoặc chiến dịch thống kê cuối cùng. Khi đánh giá bài báo phải dùng các dải seed tách biệt, các quỹ đạo chung và quy trình thống kê theo hướng dẫn; không dùng riêng lần chạy 200 slot này làm kết quả cuối cùng.
- Chưa kiểm toán độc lập toàn bộ S1/S2. Các acceptance output cũ trong gói thuộc các lần chạy S1/S2 riêng.
- Effective SNR trong phần chẩn đoán S2 được code S2 ghi PROVISIONAL; đường chạy S3 không gọi công thức đó, mà dùng gross SE từ S1.
- Chữ ký cache S2 hiện không gồm center_dir_min_deg và center_dir_max_deg. Nếu đổi các tham số này, dùng thư mục cache khác để tránh tái sử dụng quỹ đạo cũ. Không thay đổi code cache S2 trong phần S3.
- Trong PDF (1) được đối chiếu, (34)–(37) lần lượt là beam gain, proxy, EM score và lựa chọn EM. Không sử dụng cảnh báo đánh số của phiên bản PDF khác làm kết luận cho bản này.
- Nếu cần thay đổi phương trình hoặc quy ước mô hình, ghi rõ thay đổi và lý do để báo cáo thầy trước chiến dịch mô phỏng tiếp theo.
