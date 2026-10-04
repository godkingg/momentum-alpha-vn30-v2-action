# Momentum Factors + ML Alpha + Exit Rules + Cost Model — VN30 Universe

> ## Bản v2 — engine mới + kết quả THẬT (đã chạy lại notebook 01–06)
> - **Engine cũ có look-ahead cùng ngày** (vị thế vào/ra ở close `t` vẫn ăn/né return ngày `t`) và walk-forward XGBoost thiếu purge nhãn → đã thay bằng
>   **engine v2** (`src/backtest.py`, signal chốt close `t`, vị thế ăn return từ `t+1`). Đo trên dữ liệu thật: engine cũ **thổi phồng Sharpe 0.35–0.96**.
> - **Không chiến lược nào vượt 1/N có ý nghĩa thống kê** (t(excess) = +0.11 / +0.97 / +1.13, ngưỡng ≥ 2), và **thứ hạng đảo lộn khi thêm 9 tháng dữ liệu 2026**
>   (Composite: Sharpe 2.33 → 1.16, từ hạng 1 xuống dưới cả 1/N).
> - **Mô hình mặc định đổi từ Composite sang Single Factor Momentum 6-1** — chọn theo tính đơn giản + drawdown + chi phí, **không** vì Sharpe cao hơn có ý nghĩa
>   (xem mục "Mô hình mặc định"). Quay lại mô hình cũ: `--signal composite`.
> - Mọi Sharpe cũ (2.71 / 2.68 / 2.64 / 2.38...) là số của engine cũ → mục D, **KHÔNG trích dẫn**.
> Chi tiết: `reports/research_note.md` mục 11–12 · `CHANGELOG_v2.md`.

## Mục tiêu

**Ngày 18**: Kiểm tra pattern "winner tiếp tục thắng, loser tiếp tục thua"
(momentum classical) trên universe VN30-tương tự bằng 2 biến thể momentum
(6-1 và 12-1, bỏ tháng gần nhất tránh short-term reversal).

**Ngày 19**: So sánh 3 cách kết hợp nhiều factor thành 1 alpha score
(IC-Weighted / Ridge / XGBoost), dùng SHAP để xác nhận feature nào thực sự
drive prediction, đánh giá bằng IC (không chỉ RMSE).

**Ngày 20**: Test lại chiến lược Momentum + Exit Rules với **transaction
cost + slippage** — giữ nguyên giả định cost đã dùng ở project
`momentum-ridge-vn30` (0.15% + 0.08% = 0.23%/lệnh) — câu hỏi Ngày 19: Sharpe có còn
đứng vững sau chi phí giao dịch không? Kết luận cũ ("có, thắng 1/N rõ ràng") được tính bằng engine cũ. **Với engine v2: ở mẫu 2022–2025 Sharpe net vẫn cao hơn 1/N (1.87 vs 1.49) nhưng chênh lệch
nằm trong sai số chuẩn (SE 0.58), và khi thêm dữ liệu 2026 thì Composite thấp hơn 1/N (1.16 vs 1.35).**

**Bản v2**: sửa look-ahead cùng ngày của engine backtest + purge nhãn của walk-forward, thêm test chống look-ahead,
chạy lại toàn bộ trên dữ liệu thật (notebook 06 + 01–05), và đổi mô hình mặc định (xem "Mô hình mặc định").

## Dataset

- Nguồn: `vnstock` (source KBS).
- Universe: 30 mã VN30-tương tự → lọc coverage (≥95%) còn **28 mã** (loại
  TCX, VPL).
- Khoảng thời gian: **notebook 01–05 cố định `END = 2025-12-31`** (dataset 2023-01-06→2025-12-24 sau warm-up 252 phiên);
  **notebook 06 và `launcher.py` / `daily_report.py` lấy đến hôm nay** (lần chạy lại: 2022-01-04→2026-10-02, 1181 phiên; cửa sổ so sánh chung 2024-01-12→2026-09-25, 670 phiên).
  Thêm ~9 tháng 2026 làm **đổi hẳn kết quả** (xem mục C) nên mọi con số phải đi kèm khoảng thời gian của nó.
- Forward return label: N_FWD = 5 ngày (1 tuần giao dịch).

## Phương pháp

```
Giá đóng cửa + volume (28 mã)
  → Momentum_6_1, Momentum_12_1, Volume_Ratio, TS_Momentum
  → Cross-sectional rank (feature_engineering.py)
  → IC Analysis, Purged K-Fold OOS (ic_analysis.py)
  → So sánh: Best single factor / IC-Weighted / Ridge / XGBoost (ic_analysis.py + model.py)
  → SHAP TreeExplainer trên XGBoost — so sánh với feature_importances_ chuẩn
  → Tín hiệu: Single Factor Momentum_6_1 (mặc định) hoặc Composite IC-Weighted (--signal composite)
  → PositionManager (6 exit rule) → Backtest
  → Backtest engine v2 (độ trễ 1 ngày) CÓ transaction cost + slippage (0.23%/lệnh)
  → So sánh: Net vs Gross vs 1/N Equal-Weight, kèm Sharpe SE và t-stat return vượt trội
```

⚠️ **`launcher.py` và `daily_report.py` (production, dashboard + báo cáo Discord) mặc định dùng Single Factor Momentum 6-1**, KHÔNG dùng Ridge/XGBoost/Composite
(`--signal composite` để dùng mô hình cũ). ML (Ridge, XGBoost, SHAP) vẫn có trong `src/model.py` và notebook 02/05 ở vai trò nghiên cứu/so sánh.

## Kết quả chính

Tất cả Sharpe dưới đây là **net cost 0.23%/lệnh, engine v2**, đo trên dữ liệu thật vnstock, 28 mã. `Sharpe SE ≈ 0.6` ⇒ khoảng tin cậy 95% ≈ **±1.2**:
chênh lệch Sharpe nhỏ hơn ~1.2 giữa hai chiến lược không phân biệt được với nhiễu. Dùng `t(excess vs 1/N)` (kiểm định ghép cặp) để so với benchmark.

### A. IC (Purged K-Fold OOF) — thêm dữ liệu 2026 làm IC của Momentum_6_1 giảm một nửa

| Mean IC | Dữ liệu đến 2025-12 (notebook 01–05) | Dữ liệu đến 2026-10 (notebook 06) |
| --- | --- | --- |
| Momentum_6_1 | 0.0525 | **0.0241** |
| Momentum_12_1 | 0.0441 | 0.0335 |
| TS_Momentum | 0.0335 | 0.0296 |
| Volume_Ratio | 0.0058 | 0.0104 |
| Composite (IC-weighted) | 0.0467 – 0.0570 (tuỳ bộ feature) | — |
| Ridge / XGBoost (K-fold) | 0.0172 / 0.0377 | — |
| XGBoost walk-forward (purged) | 0.0207 | 0.0223 (Rank IC, cửa sổ 2024-01→2026-09) |

Rank IC trong cửa sổ walk-forward 2024-01→2026-09 (notebook 06): **Composite 0.0442 (t điều chỉnh 1.87)** · XGBoost 0.0223 (1.17) · Single 0.0176 (0.72) — tức Composite có IC cao nhất
nhưng Sharpe thấp nhất ở mục B: **IC cao hơn không đồng nghĩa Sharpe cao hơn** (bài học giữ nguyên). Khớp với Ngày 25: IC các model ≈ 0 và âm ở fold cuối (2025-12→2026-09).

### B. Kết quả engine v2 — cửa sổ chung 2024-01-12 → 2026-09-25 (670 phiên), net cost

| Chiến lược | Sharpe ± 1.96·SE | Ann Return | Max DD | Calmar | Win rate | Rank IC (t) | **t(excess vs 1/N)** | Cost đã trả | BUY | Hold TB |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| IC-Weighted Composite | 1.16 ± 1.20 | 24.8% | −23.4% | 1.06 | 48.9% | 0.0442 (1.87) | **+0.11** | 10.6% | 290 | 41 ngày |
| **Single Factor (Mom 6-1)** | 1.49 ± 1.20 | 31.4% | −18.9% | 1.66 | 52.3% | 0.0176 (0.72) | +0.97 | 8.6% | 289 | 46 ngày |
| XGBoost walk-forward (purged) | 1.51 ± 1.20 | 34.3% | −20.7% | 1.66 | 49.1% | 0.0223 (1.17) | +1.13 | **28.1%** | 442 | 15 ngày |
| 1/N Equal-Weight | 1.35 ± 1.20 | 24.8% | **−16.8%** | 1.48 | — | — | — | — | — | — |

- **Không chiến lược chủ động nào vượt 1/N có ý nghĩa thống kê** (|t| < 2). Composite thấp hơn cả 1/N.
- **Cost kéo Sharpe xuống** (gross → net): Composite 1.41 → 1.16 (−0.25) · Single 1.71 → 1.49 (−0.22) · **XGBoost 2.18 → 1.51 (−0.67)** — XGBoost giao dịch gấp ~3 lần (307 `SIGNAL_EXIT`).
- Exit rule chiếm nhiều nhất: Composite `MOM_FLIP` 122, `HARD_STOP` 65, `TRAILING_STOP` 55 (≈ 42% là stop) — chính vì stop kích hoạt thường xuyên nên lỗi look-ahead của engine cũ gây lệch lớn (mục D).

### C. Thứ hạng KHÔNG ổn định theo giai đoạn (cùng engine v2, cùng điểm bắt đầu)

| Cửa sổ | Composite | Single | XGBoost WF | 1/N | Sharpe SE |
| --- | --- | --- | --- | --- | --- |
| 2022→2025-12 (notebook 03/04; cửa sổ chung Composite & Single, sau warm-up) | 1.87 | 1.69 | — | 1.49 | 0.58 |
| 2024-01→2025-12 (notebook 05, 488 phiên) | **2.33** (hạng 1) | 2.29 | 1.72 | 1.80 | 0.72 |
| 2024-01→2026-09 (notebook 06, 670 phiên) | **1.16** (hạng cuối) | 1.49 | 1.51 | 1.35 | 0.61 |

Thêm ~183 phiên (25/12/2025 → 25/09/2026) làm mọi Sharpe giảm. Ước lượng gián tiếp từ tổng return của hai lần chạy (chỉ hợp lệ với chiến lược không có tham số ước lượng
trên toàn mẫu): **Single ≈ −3%, XGBoost ≈ +18%, 1/N ≈ +4%** trong giai đoạn đó; Composite không so sánh được vì trọng số IC được ước lượng lại khi thêm dữ liệu.
→ Kết luận "mô hình X tốt nhất" **đổi dấu theo 9 tháng dữ liệu**; đây là bằng chứng cho thấy các chênh lệch Sharpe là nhiễu/chế độ thị trường, không phải khác biệt chất lượng ổn định.

### D. Engine cũ thổi phồng Sharpe bao nhiêu? (đo trên dữ liệu thật; số cũ → số v2)

| Mẫu | Composite | Single | XGBoost WF | 1/N |
| --- | --- | --- | --- | --- |
| 2022→2025-12 (notebook 03/04) | 2.38 → 1.87 (−0.50) | 2.17 → 1.69 (−0.48) | — | 1.50 → 1.49 (≈0) |
| 2024-01→2025-12 (notebook 05) | 2.71 → 2.33 (−0.39) | 2.64 → 2.29 (−0.35) | 2.68 → 1.72 (**−0.96**) | 1.80 → 1.80 (≈0) |
| 2024-01→2026-09 (notebook 06, cùng dữ liệu, so trực tiếp) | 1.94 → 1.16 (−0.78) | 2.18 → 1.49 (−0.69) | 2.06 → 1.51 (−0.55) | — |

- Engine cũ **chỉ thổi phồng chiến lược chủ động** (1/N không có stop nên không đổi) — đúng với cơ chế lỗi. Mức thổi phồng 0.35–0.96 ≈ 0.6–1.6 Sharpe SE.
- XGBoost chịu thêm lỗi thiếu purge nên lệch lớn nhất (−0.96 ở notebook 05).
- ⚠️ **Đính chính ước lượng trước đó**: thử nghiệm dữ liệu GIẢ LẬP cho thấy chỉ ~0.1–0.2 với stop mặc định — **ước lượng đó quá thấp**. Dữ liệu momentum thật làm stop kích hoạt nhiều hơn hẳn, nên mức lệch thật lớn hơn 3–5 lần.

### E. Mô hình mặc định: Single Factor Momentum 6-1 (đổi từ Composite)

**Trung thực trước: không có mô hình nào được chứng minh là tốt nhất.** Sharpe giữa ba chiến lược chênh nhau ≤ 0.35 trong khi SE ≈ 0.6, không ai vượt 1/N có ý nghĩa, và thứ hạng đảo khi thêm dữ liệu.
Vì vậy mặc định được chọn theo các tiêu chí **ổn định qua cả hai cửa sổ**, không theo Sharpe:

| Tiêu chí | Composite | **Single (Mom 6-1)** | XGBoost WF | 1/N |
| --- | --- | --- | --- | --- |
| Calmar (2024→2025-12 / 2024→2026-09) | 2.71 / 1.06 | **2.96 / 1.66** | 2.10 / 1.66 | 1.97 / 1.48 |
| Max DD (2 cửa sổ) | −18.7% / −23.4% | **−16.3% / −18.9%** | −18.1% / −20.7% | −16.8% / −16.8% |
| Cost đã trả (670 phiên) | 10.6% | 8.6% | 28.1% | — |
| Tham số ước lượng từ dữ liệu | 4 trọng số IC (**in-sample**) | **0** | cả mô hình, refit mỗi 21 phiên | 0 |
| IC Purged K-Fold 2022–2025 (notebook 02) | 0.0467 | **0.0500** | 0.0377 | — |

Lý do chọn Single: (1) Calmar đứng đầu/đồng hạng đầu và Max DD tốt nhất trong các chiến lược chủ động ở cả hai cửa sổ; (2) cost thấp nhất; (3) **không có tham số nào ước lượng từ dữ liệu** →
hết thiên lệch in-sample của trọng số IC, không cần `state/ic_weights.json`, không cần refit/đóng băng; (4) notebook 02 (IC out-of-fold) cũng cho Single ≥ Composite/Ridge/XGBoost.
Không chọn XGBoost: Sharpe danh nghĩa cao nhất ở cửa sổ mới nhưng thấp nhất ở cửa sổ cũ, cost 28%, giữ vị thế ~15 ngày, phức tạp vận hành.

**Giới hạn của lựa chọn này**: chọn sau khi quan sát hai cửa sổ chồng lấn nhau ⇒ có data-snooping; hãy coi đây là *mặc định cho paper trading*, không phải bằng chứng có alpha.
Nếu mục tiêu là độ tin cậy cao nhất, **1/N** (Sharpe 1.35, Max DD −16.8%, không phí vận hành) là phương án thay thế hợp lý — chiến lược chủ động **chưa chứng minh được giá trị gia tăng sau cost**.
Quay lại Composite: `python launcher.py --signal composite` · `python daily_report.py --mode final --signal composite`.
Hệ quả vận hành: đổi mặc định làm danh mục mô hình của báo cáo hằng ngày tính lại hoàn toàn — lần chạy đầu sẽ cảnh báo "Đổi mô hình tín hiệu" và bỏ qua đối chiếu sổ lệnh cũ.

### F. Bằng chứng kiểm chứng engine (không cần dữ liệu thật)

| Kiểm tra | Kết quả |
| --- | --- |
| Vị thế mở ở close `t` không ăn return ngày `t` (kịch bản tất định) | ✅ engine v2 = 0; engine cũ ăn +10% |
| Cú giảm kích hoạt stop vào PnL ngày `t` | ✅ engine v2 ghi −20%; engine cũ né = 0 |
| Signal = return ngày mai (nhìn trước) vs return hôm nay | ✅ Sharpe > 10 vs < 1/10 mức đó (engine không lệch ngày) |
| `walk_forward_signal` bất biến khi xáo nhãn tương lai | ✅ qua với `n_fwd=5`; **bắt được lỗi** với `n_fwd=0` |
| `launcher.py` / `daily_report.py` chạy end-to-end (dữ liệu giả lập) + JSON dashboard | ✅ `tests/test_launcher_smoke.py`, `tests/test_daily_report.py` |

### G. Kết quả CŨ — engine có look-ahead (⚠️ chỉ tham khảo, KHÔNG trích dẫn)

Sharpe có cost: Composite 2.38 / 2.71 · Single 2.17 / 2.64 · XGBoost 2.68 · 1/N 1.50 / 1.80 (notebook 04 / 05 cũ). Bản notebook cũ còn nguyên output ở `notebooks/legacy_v1_outputs/`.
Kết luận cũ "Composite thắng rõ ràng, thắng 1/N rõ ràng" **không còn đúng**.

## Báo cáo tự động hằng ngày lên Discord (miễn phí · không VPS · không cần máy local)

Mỗi ngày giao dịch (thứ 2–6, giờ Việt Nam), GitHub Actions tự chạy và gửi vào kênh Discord của bạn **ảnh PNG + file HTML**:

| Giờ | Chế độ | Nội dung |
| --- | --- | --- |
| **14:00** | `preview` — lệnh dự kiến | Danh sách MUA/BÁN/GIẢM theo giá gần nhất (nến chưa chốt, còn kịp đặt lệnh ATC 14:30–14:45), vị thế đang giữ kèm **giá cắt lỗ / trailing** để đặt lệnh điều kiện, watchlist top rank |
| **15:00** | `final` — cuối phiên | Lệnh chốt, return ngày (net cost) so với 1/N, lũy kế, drawdown, và **so sánh với lệnh dự kiến 14:00** (mới / không còn / giữ nguyên) |

Cách hoạt động: Discord **webhook** (chỉ cần 1 URL, không cần bot online 24/7) + **GitHub Actions cron** (miễn phí; repo public không giới hạn phút, repo private ~2.000 phút/tháng, mỗi ngày dùng ≈ 25 phút).
PNG vẽ bằng matplotlib (không cần trình duyệt); HTML là file độc lập (không CDN/JS) — Discord không xem trước HTML nên nó là file đính kèm để tải về mở bằng trình duyệt.

### Cài đặt (≈10 phút)
1. **Tạo webhook**: Discord → Server Settings → Integrations → Webhooks → New Webhook → chọn kênh → *Copy Webhook URL*.
2. **Đẩy repo này lên GitHub** (thư mục `.github/workflows/` phải nằm trong repo).
3. **Lưu bí mật**: GitHub repo → Settings → Secrets and variables → Actions → *New repository secret* → tên `DISCORD_WEBHOOK_URL`, giá trị là URL vừa copy.
   (Tùy chọn: tab *Variables* → `DISCORD_MENTION` = `<@ID_Discord_của_bạn>` để được ping.)
4. **Chạy thử ngay** (đừng chờ tới 14:00): tab *Actions* → *Daily report (Discord)* → *Run workflow* → `mode=final`.
   Chạy vào cuối tuần/nghỉ lễ thì điền `asof` = ngày giao dịch gần nhất (vd `2026-10-02`) — chạy kiểu này **không commit state**.
5. Kiểm tra 3 điều ở lần chạy thử đầu: (a) bước tải dữ liệu **không bị chặn IP** của GitHub; (b) nến hôm nay **có xuất hiện lúc 14:00** (chạy `mode=preview` trong giờ giao dịch); (c) ảnh trong Discord hiển thị đúng.
   Từ đó lịch tự chạy; mỗi lần có thêm artifact (ảnh/HTML/JSON) tải về được trong tab *Actions* (giữ 14 ngày).

### Lịch chạy và độ trễ
| Cron (UTC) | Giờ VN | Việc |
| --- | --- | --- |
| `50 6 * * 1-5` | 13:50 | khởi động, **tự chờ tới đúng 14:00** rồi tải dữ liệu + gửi preview |
| `50 7 * * 1-5` | 14:50 | khởi động, **tự chờ tới đúng 15:00** rồi gửi báo cáo cuối phiên |

GitHub **không đảm bảo đúng giờ**: cron thường trễ 5–30 phút lúc cao điểm, hiếm khi bị bỏ qua hẳn. Lịch được đặt sớm 10 phút để bù; nếu GitHub trễ quá giờ hẹn thì báo cáo đến muộn tương ứng
(preview trễ quá ~14:25 là không còn ý nghĩa — hãy coi đó là tín hiệu tham khảo). Việc tải 30 mã mất ~2–3 phút nên ảnh thường đến 14:03–14:05 và 15:03–15:05.

### An toàn vận hành đã dựng sẵn
- **Mô hình mặc định là Single Factor Momentum 6-1** (không có tham số nào cần đóng băng). Với `--signal composite`, trọng số IC được **đóng băng** trong `state/ic_weights.json` (tính lại mỗi ngày làm lịch sử mô phỏng đổi nhẹ → lệnh hôm qua có thể biến mất; chỉ tính lại khi chạy tay với `refit_weights=true`).
- **Đổi mô hình tín hiệu** (vd từ Composite sang Single): sổ lệnh ghi kèm tên mô hình; khác mô hình thì báo cáo đầu tiên cảnh báo "Đổi mô hình tín hiệu" và **bỏ qua** đối chiếu sổ lệnh cũ (thay vì báo lệch giả).
- **Chặn universe lệch**: thiếu/thừa mã so với lần trước (vd một mã tải lỗi) → script thử tải lại 3 lần rồi **dừng và báo lỗi lên Discord**, không phát lệnh trên dữ liệu thiếu.
- **Sổ lệnh** (`state/ledger.json`): mỗi phiên tính lại danh mục hôm qua và so với sổ đã lưu; lệch → cảnh báo ngay trong báo cáo (thường do nguồn dữ liệu sửa giá lịch sử, vd điều chỉnh cổ tức).
- **Cuối tuần** bỏ qua im lặng; **ngày lễ / chưa có nến** → gửi 1 dòng thông báo (tắt bằng `--no-notice`). Lỗi bất kỳ → báo lên Discord (không lộ URL webhook) và job báo đỏ để GitHub gửi mail.
- Commit `state/` mỗi ngày giao dịch cũng giữ repo luôn "có hoạt động" — GitHub tự tắt workflow theo lịch sau 60 ngày repo không có hoạt động.

### Giới hạn cần biết
- Đây là **danh mục mô hình (paper)**: giá khớp giả định = close, tỷ trọng equal-weight 1/số vị thế, chưa có T+2/thanh khoản. Báo cáo **không phải khuyến nghị đầu tư**, và mục *Kết quả engine v2* ở trên vẫn chưa có số thật — hãy chạy notebook 06 trước khi dựa vào tín hiệu.
- Lúc 14:00 **khối lượng hôm nay chưa đủ phiên** nên factor Volume_Ratio bị thấp hơn thực tế → thứ hạng có thể đổi lúc 15:00 (báo cáo cuối phiên nêu rõ lệnh nào thay đổi).
- Lịch sử mô phỏng bắt đầu từ `--start 2022-01-01` và **nhịp rebalance 5 phiên phụ thuộc vị trí ngày trong chuỗi dữ liệu** → đừng đổi `--start` giữa chừng (sẽ lệch toàn bộ lịch rebalance).
- Chiến lược có thể **bán rồi mua lại cùng mã trong cùng phiên** (thoát theo stop/rule nhưng rank vẫn cao): báo cáo tự cảnh báo; khi đặt lệnh thật có thể bỏ cả hai.
- Chưa kiểm chứng được trong môi trường tạo bản này: (1) GitHub có bị nguồn dữ liệu chặn IP không; (2) nến intraday lúc 14:00 có sẵn không → bước 5 ở trên.

Chạy tay trên máy: `python daily_report.py --mode final --no-discord --asof 2026-10-02` (sinh file vào `reports/daily/`).

## Lệnh slash `/report` — xem báo cáo bất cứ lúc nào (miễn phí · không VPS · không cần máy local)

```
Bạn gõ /report ──► Cloudflare Worker (free, luôn online) ──► GitHub Actions ──► bot đăng PNG + HTML vào ĐÚNG KÊNH bạn gõ lệnh
                   xác thực chữ ký, phân quyền, chống spam    tạo báo cáo (3–5 phút)
```

| Bạn gõ | Kết quả |
| --- | --- |
| `/report` lúc **đang trong phiên** (T2–T6, 09:00–15:00) | **Lệnh tạm tính** theo giá gần nhất (nến chưa chốt) |
| `/report` lúc **sau 15:00 / cuối tuần / nghỉ lễ** | **Báo cáo cuối phiên** của phiên giao dịch gần nhất (có ghi chú nếu hôm nay không có phiên) |
| `/report date:02/10/2026` (hoặc `2026-10-02`) | **Báo cáo cuối phiên** của ngày đó (phải là ngày có phiên, trong quá khứ) |

Vì sao cần Worker: Discord phải gửi lệnh tới một địa chỉ HTTPS **luôn sẵn sàng** (GitHub Actions chỉ chạy theo lịch/khi được gọi nên không làm được). Cloudflare Workers gói free (100.000 request/ngày) làm đúng việc đó — và không có máy nào của bạn phải bật.
Bot **không cần online** (chỉ dùng REST để đăng tin), nên không phải giữ tiến trình `discord.py` chạy như bot cũ.

### Cài đặt một lần (≈20 phút) — nên làm SAU khi báo cáo theo lịch đã chạy ít nhất 1 lần (để có sổ lệnh `state/ledger.json`; riêng `--signal composite` còn cần `state/ic_weights.json`)
1. **Discord Developer Portal** → *New Application*. Ở *General Information* copy **Application ID** và **Public Key**. Ở tab *Bot* → *Reset Token* → copy **Bot Token** (không cần bật Privileged Intents).
2. **Mời bot vào server** bằng link (thay `APP_ID`): `https://discord.com/oauth2/authorize?client_id=APP_ID&scope=bot%20applications.commands&permissions=52224`
   (quyền: xem kênh, gửi tin, nhúng link, đính kèm file). Bot phải có quyền ở **kênh bạn sẽ gõ `/report`**.
3. **GitHub**: thêm Secret `DISCORD_BOT_TOKEN`. Tạo **fine-grained PAT** (Settings → Developer settings → Fine-grained tokens): *Only select repositories* → repo này, quyền **Actions: Read and write**, đặt hạn dùng. Token này chỉ dùng ở bước 4.
4. **Cloudflare** (tài khoản free): *Workers & Pages → Create → Hello World → Edit code* → dán toàn bộ `worker/worker.js` → *Deploy*. Rồi *Settings → Variables and Secrets*:

   | Tên | Loại | Giá trị |
   | --- | --- | --- |
   | `DISCORD_PUBLIC_KEY` | Variable | Public Key ở bước 1 |
   | `GH_REPO` | Variable | `owner/repo` |
   | `GH_TOKEN` | **Secret** | PAT ở bước 3 |
   | `ALLOWED_GUILD_IDS` | Variable | ID server của bạn (**nên đặt**; bật *Developer Mode* → chuột phải server → Copy ID) |
   | `ALLOWED_ROLE_IDS` / `ALLOWED_USER_IDS` | Variable, tuỳ chọn | giới hạn người được dùng (cách nhau dấu phẩy) |
   | `COOLDOWN` | **KV binding** | tạo KV namespace rồi gắn vào Worker (**nên có**: chống spam, mặc định 3 phút/người, 45 giây toàn server) |

5. Developer Portal → *General Information* → **Interactions Endpoint URL** = địa chỉ Worker (`https://….workers.dev`) → *Save* (Discord tự gửi PING kiểm tra chữ ký; báo lỗi nghĩa là sai Public Key).
6. **Đăng ký lệnh** (chạy 1 lần ở bất kỳ đâu, kể cả Colab: `!pip -q install requests` rồi chạy lệnh dưới):
   `DISCORD_BOT_TOKEN=... python scripts/register_discord_commands.py --app-id APP_ID --guild-id SERVER_ID`
7. Vào kênh, gõ `/report`. Thấy "⏳ Đã nhận yêu cầu…" rồi 3–5 phút sau là báo cáo (bot ping bạn).

### Giới hạn và bảo mật cần biết
- **Không tức thời**: GitHub Actions khởi động + tải 30 mã mất **3–5 phút** (không thể nhanh hơn nếu không có server chạy liên tục). Nếu cần < 10 giây thì phải có máy luôn bật.
- **Chống lạm dụng**: ai gõ được `/report` đều tốn phút Actions và hạn mức vnstock. Hãy đặt `ALLOWED_GUILD_IDS`, cân nhắc `ALLOWED_ROLE_IDS`, và gắn KV. KV nhất quán sau cùng (trễ tới ~1 phút giữa các điểm biên) nên cooldown là chống spam *tương đối*, không phải khoá chặt.
- **Bí mật**: `GH_TOKEN` chỉ nằm ở Worker (Secret) và có quyền tối thiểu (chỉ Actions của 1 repo); `DISCORD_BOT_TOKEN` chỉ ở GitHub Secrets; **không token nào đi qua tham số workflow** (nên không hiện trong tab Actions, kể cả repo public). Worker xác thực chữ ký Ed25519 của Discord và từ chối request quá hạn > 5 phút (chống replay). Ngày nhập vào được kiểm tra ở Worker và ở script, workflow truyền qua `env:` (chống chèn lệnh).
- **PAT hết hạn** thì `/report` báo "GitHub trả về 401" — tạo PAT mới và cập nhật `GH_TOKEN`.
- **Báo cáo theo yêu cầu không ghi `state/`** (không đổi sổ lệnh/trọng số của báo cáo theo lịch). Với `--signal composite` mà chưa có trọng số đóng băng, báo cáo sẽ ghi rõ cảnh báo.
- **Báo cáo ngày cũ là mô phỏng lại bằng dữ liệu hiện có** (trọng số IC đóng băng hôm nay), có thể khác với những gì báo cáo ngày đó đã gửi nếu nguồn dữ liệu điều chỉnh giá lịch sử.
- Bot không đăng được vào kênh (thiếu quyền) → báo cáo được gửi vào kênh của webhook mặc định thay thế.

## Hạn chế & Rủi ro

- **Không có mô hình nào vượt 1/N có ý nghĩa thống kê** (t(excess) ≤ 1.13) và thứ hạng đảo theo giai đoạn (mục B, C). Mô hình mặc định (Single) là lựa chọn theo
  đơn giản/drawdown/cost, đã có data-snooping nhẹ — xem mục E.
- **Trọng số IC của Composite tính trên toàn mẫu** rồi backtest chính mẫu đó (thiên lệch in-sample) — một lý do nữa khiến Single (0 tham số) được ưu tiên;
  nếu dùng Composite lâu dài phải tính trọng số expanding-window.
- **Khớp lệnh tại close** ngày tín hiệu: không T+2, không giới hạn thanh khoản/market impact.
- **Benchmark 1/N** rebalance hằng ngày và không cost (hơi có lợi cho 1/N — thận trọng với chiến lược).
- **Cửa sổ out-of-sample chỉ ~2.7 năm** (1 chế độ bull + 1 giai đoạn điều chỉnh 2026), Sharpe SE ≈ 0.6 — không đủ để xếp hạng mô hình.
- **Chưa có out-of-sample tách biệt** — tham số PositionManager
  (stop_loss=8%, trail=12%, top_entry_rank=0.65...) được set cố định, chưa
  test trên giai đoạn hoàn toàn chưa thấy để kiểm tra overfitting tham số.
- **Long-short TS-Momentum kém hiệu quả** do bull bias của thị trường VN —
  Short leg kéo hiệu suất xuống. Momentum long-short kinh điển (kiểu Mỹ)
  KHÔNG hoạt động tốt trực tiếp trên VN, cần chuyển sang long-only.
- **Ridge (K-fold) thua Momentum_6_1 đứng riêng** trên universe 28 mã (IC, không phụ thuộc engine).
  XGBoost walk-forward giao dịch nhiều hơn đáng kể (442 buy vs 290 của Composite, cost 28% vs 10.6%); cần universe lớn hơn
  và market-impact model trước khi đánh giá lợi thế triển khai.
- **Đa cộng tuyến giữa Momentum_6_1 và Momentum_12_1** — Ridge coefficient
  của Momentum_12_1 gần như 0 dù IC riêng lẻ dương, dấu hiệu 2 factor tương
  quan cao (Pearson corr ~0.72, cùng công thức, chỉ khác lookback).
- **SHAP trong `notebooks/02_ml_alpha_prediction.ipynb` là phần MỞ RỘNG**
  theo yêu cầu — notebook gốc `day19_test.ipynb` không chứa SHAP. Đã chạy
  thành công end-to-end (bao gồm `shap.TreeExplainer`, summary plot,
  beeswarm plot) — xem output trong notebook đã upload gần nhất.
- **Cost model giả định đơn giản** — 0.23%/lệnh cố định, KHÔNG mô hình hóa
  market impact tăng theo kích thước lệnh, KHÔNG phân biệt cost mua vs bán
  (thuế bán 0.1% ở VN thường cao hơn mua), KHÔNG tính spread thay đổi theo
  thanh khoản từng mã. Đây là xấp xỉ hợp lý cho nghiên cứu ban đầu, chưa
  phải mô hình cost sát thực tế broker cụ thể.
- **Universe hẹp, VN30 được xác định tại ngày chạy** — 28 mã tự chọn, thiên về
  ngân hàng/blue-chip (tương lai có thể thay đổi).

## Cách chạy

```bash
# 1. Cài dependencies
pip install -r requirements.txt

# 2. (Tùy chọn) Dashboard mẫu với snapshot ENGINE CŨ (có banner cảnh báo) — đã sinh sẵn trong zip
python scripts_bootstrap_old_numbers.py

# 3. Chạy 5 notebook (Ngày 18 → 20) để tái tạo phân tích với dữ liệu vnstock hiện tại
jupyter notebook notebooks/01_momentum_factors.ipynb        # Momentum factors, IC, quintile
jupyter notebook notebooks/02_ml_alpha_prediction.ipynb     # Feature matrix, Ridge/XGBoost, SHAP
jupyter notebook notebooks/03_backtest_with_costs.ipynb     # Backtest CÓ transaction cost + slippage
jupyter notebook notebooks/04_single_factor_backtest.ipynb  # Single Factor vs Composite (có cost)
jupyter notebook notebooks/05_xgboost_walkforward_backtest.ipynb  # XGBoost walk-forward (purged) vs Single/Composite
jupyter notebook notebooks/06_engine_v2_rerun.ipynb          # ★ engine cũ vs v2, bảng kết quả thật (bản đi kèm đã có output của lần chạy 2026-10-02)

# 4. Chạy pipeline đầy đủ + backtest (mặc định CÓ cost) + dashboard
python launcher.py                      # có cost 0.23%/lệnh, tự mở dashboard sau khi xong
python launcher.py --no-cost            # backtest KHÔNG cost (tái tạo baseline Ngày 19)
python launcher.py --no-open            # chỉ tính toán, không tự mở trình duyệt
python launcher.py --compare-legacy     # chạy thêm engine CŨ trên cùng dữ liệu để đo mức Sharpe bị thổi phồng

# 5. Chạy test (look-ahead engine / purge walk-forward / purged K-fold / entry-exit rule / cost model / launcher smoke)
pytest tests/ -v
```

Sau khi chạy `launcher.py`, mở `dashboard/index.html` (hành động hôm nay:
BUY/AVOID/REDUCE/EXIT/NEUTRAL cho từng mã) và `dashboard/backtest.html`
(metrics đầy đủ NET vs GROSS vs 1/N, trade log, exit breakdown) trực tiếp
bằng trình duyệt — không cần server, dữ liệu được nhúng sẵn vào
`dashboard/embed_data.js`.

**Encoding**: toàn bộ file Python/Markdown trong project dùng UTF-8
(`# -*- coding: utf-8 -*-` ở đầu mỗi file `.py`), JSON ghi với
`ensure_ascii=False` để giữ nguyên tiếng Việt có dấu, tránh lỗi mojibake.

## Cấu trúc project

```
momentum-ml-alpha-vn30/
├── README.md
├── requirements.txt
├── daily_report.py                    # ★ báo cáo hằng ngày → Discord (PNG + HTML), chạy bằng GitHub Actions
├── requirements-report.txt            # thư viện tối thiểu cho báo cáo hằng ngày
├── .github/workflows/daily_report.yml # lịch 14:00 + 15:00 giờ VN
├── .github/workflows/report_on_demand.yml # chạy khi có lệnh /report (Worker kích hoạt)
├── worker/                            # Cloudflare Worker nhận slash command (worker.js + test Node)
├── scripts/register_discord_commands.py # đăng ký /report với Discord (1 lần)
├── state/                             # trọng số IC đóng băng, sổ lệnh, nhật ký (workflow tự commit)
├── launcher.py                        # orchestrator: data → factors → tín hiệu (--signal single|composite) → backtest → dashboard JSON
├── scripts_bootstrap_old_numbers.py   # sinh dashboard bằng số THẬT từ day19_test.ipynb
├── data/                              # (rỗng — vnstock tải trực tiếp, không cache)
├── notebooks/
│   ├── 01_momentum_factors.ipynb      # Ngày 18: Mom_6_1 vs Mom_12_1, IC, quintile analysis
│   ├── 02_ml_alpha_prediction.ipynb   # Ngày 19: feature matrix, Ridge/XGBoost, SHAP, IC evaluation
│   ├── 03_backtest_with_costs.ipynb   # Ngày 20: backtest CÓ transaction cost + slippage (0.23%/lệnh)
│   ├── 04_single_factor_backtest.ipynb    # Ngày 21: Single Factor Momentum_6_1 vs Composite (có cost)
│   ├── 05_xgboost_walkforward_backtest.ipynb  # Ngày 22: XGBoost walk-forward (purged) vs Single/Composite/1N
│   ├── 06_engine_v2_rerun.ipynb       # ★ v2: chạy lại toàn bộ bằng engine mới + đo Δ so với engine cũ
│   └── legacy_v1_outputs/             # bản notebook 03-05 CŨ còn nguyên output (engine có look-ahead) — chỉ để đối chiếu
├── src/
│   ├── data_loader.py                 # vnstock, filter coverage
│   ├── momentum_factors.py            # Ngày 18: Mom_6_1, Mom_12_1, quintile analysis
│   ├── ic_analysis.py                 # Module IC/IC-IR dùng chung (Purged K-Fold) + composite_score_ic
│   ├── feature_engineering.py         # Ngày 19: cross-sectional rank, feature matrix, label
│   ├── model.py                       # Ngày 19: Linear/Ridge, XGBoost, SHAP + walk_forward_signal (Ngày 22; v2: purge n_fwd)
│   ├── strategy.py                    # Composite score, PositionManager (entry/exit rules)
│   ├── pipeline.py                    # dựng tín hiệu dùng chung (single mặc định / composite) cho launcher + báo cáo hằng ngày
│   ├── backtest.py                    # ★ ENGINE v2 (độ trễ 1 ngày): backtest_with_exits, invested_returns, port_metrics, sharpe_se
│   ├── backtest_legacy.py             # engine CŨ (look-ahead) — chỉ để đối chứng/đo chênh, KHÔNG dùng cho kết quả
│   └── risk.py                        # Benchmark, compare_strategies (+Sharpe SE), excess_return_tstat
├── reports/
│   └── research_note.md               # Observation/Hypothesis/Evidence/Conclusion, chuẩn Ngày 27
├── tests/
│   ├── test_no_lookahead.py
│   ├── test_purged_kfold.py
│   ├── test_feature_engineering.py
│   ├── test_position_manager.py
│   ├── test_cost_model.py             # Ngày 20: cost_rate=0 backward-compat, cost>0 không cải thiện Sharpe
│   ├── test_composite_ic.py           # Ngày 21: composite_score_ic() đo đúng framework OOS
│   ├── test_walk_forward_signal.py    # Ngày 22: walk_forward_signal() không look-ahead (+ test purge nhãn, v2)
│   ├── test_engine_no_lookahead.py    # ★ v2: engine không ăn return ngày vào lệnh, stop-loss vào PnL, đối chứng engine cũ
│   └── test_launcher_smoke.py         # ★ v2: launcher end-to-end bằng dữ liệu giả lập
└── dashboard/
    ├── index.html                     # Hành động hôm nay: BUY/AVOID/REDUCE/EXIT/NEUTRAL
    ├── backtest.html                  # Metrics NET vs GROSS vs 1/N, trade log, cost info (+ banner phiên bản engine)
    ├── vendor_chart.js                # Chart.js offline (không cần internet)
    ├── embed_data.js                  # dữ liệu nhúng sẵn (sinh bởi launcher.py hoặc bootstrap script)
    └── data/today.json, backtest.json
```
