from vnlaw_rag.answer import _money_values, check_numbers
from vnlaw_rag.vehicle import detect_vehicles, vehicle_penalty


def test_money_parse():
    assert _money_values("phạt từ 4.000.000 đồng đến 6 triệu") == {4_000_000, 6_000_000}


def test_check_numbers_flags_hallucinated_fine():
    ctx = ["Phạt tiền từ 4.000.000 đồng đến 6.000.000 đồng ... bị trừ điểm giấy phép lái xe 4 điểm"]
    assert check_numbers("Phạt 4 triệu đến 6 triệu, trừ 4 điểm", ctx) == []
    w = check_numbers("Phạt 800.000 đồng", ctx)
    assert w and "800000" in w[0]


def test_detect_vehicles():
    assert detect_vehicles("xe may vuot den do") == {"moto"}
    assert detect_vehicles("Ô tô đỗ trên cao tốc") == {"oto"}
    assert detect_vehicles("vượt đèn đỏ") == set()
    # "ô tô" nằm trong "mô tô" -> phải so theo ranh giới từ
    assert detect_vehicles("Điều 7. Xử phạt người điều khiển xe mô tô, xe gắn máy") == {"moto"}
    assert detect_vehicles("người điều khiển xe máy chuyên dùng") == {"tractor"}


def test_vehicle_penalty():
    title = "Điều 6. Xử phạt người điều khiển xe ô tô vi phạm quy tắc giao thông"
    assert vehicle_penalty({"moto"}, title) == 0.5
    assert vehicle_penalty({"oto"}, title) == 1.0
    assert vehicle_penalty(set(), title) == 1.0
