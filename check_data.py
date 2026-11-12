
from pathlib import Path

root = Path(__file__).resolve().parent / "data"

categories = ["carpet", "bottle", "hazelnut"]

for category in categories:
    path = root / category

    train = path / "train" / "good"
    test = path / "test"
    ground_truth = path / "ground_truth"

    normal_images = list(train.glob("*.png"))
    test_images = list(test.rglob("*.png"))

    print(f"\n===== {category.upper()} =====")
    print(f"Train images: {len(normal_images)}")
    print(f"Test images: {len(test_images)}")

    if (
        train.is_dir()
        and test.is_dir()
        and ground_truth.is_dir()
        and len(normal_images) > 0
        and len(test_images) > 0
    ):
        print("PASS: 資料夾結構正確")
    else:
        print("FAIL: 請檢查資料夾或圖片")
