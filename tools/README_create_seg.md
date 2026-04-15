# create_seg.py

X-AnyLabeling JSON 라벨을 8-bit encoded semantic segmentation PNG mask로 변환하고 시각화하는 독립 스크립트.

## 의존성

```bash
pip install numpy opencv-python
```

> `--viz` 모드는 `cv2.imshow`를 사용하므로 `opencv-python` (headless 아님)이 필요하다.
> `--convert` 모드는 headless 버전에서도 동작한다.

## 디렉토리 구조

```
<root>/
    <scenario_00>/
        <subdir>/               # default: "Noon"
            CAMERA_1/
                0.jpg
                0.json          # X-AnyLabeling polygon label
                100.jpg
                ...
            CAMERA_2/
            ...
    <scenario_01>/
    ...
```

## 사용법

### 변환 (JSON -> PNG mask)

```bash
python create_seg.py --convert \
    --root /path/to/dataset_root \
    --dst  /path/to/output_dir
```

출력 파일명: `<scenario>_<camera>_<timestamp>.png`

예시: `a1_scenario_00_1_0.png` (scenario=a1_scenario_00, CAMERA_1, timestamp=0)

### 시각화

```bash
python create_seg.py --viz \
    --root /path/to/dataset_root \
    --dst  /path/to/output_dir
```

조작:
- 오른쪽 화살표: 다음 이미지
- 왼쪽 화살표: 이전 이미지
- ESC: 종료

## 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--root` | (필수) | 시나리오 폴더들이 있는 상위 디렉토리 |
| `--dst` | `<root>/seg_labels` | 변환된 mask PNG 저장 경로 |
| `--subdir` | `Noon` | 시나리오 하위 디렉토리명 |
| `--cameras` | `1,2,3,4,5` | 대상 카메라 번호 (쉼표 구분) |

## 출력 포맷

- dtype: `uint8`, shape: `(H, W)`, single-channel PNG
- pixel value 0 = background (unlabeled)
- pixel value 1~ = 클래스별 trainId

```python
import cv2
seg = cv2.imread("a1_scenario_00_1_0.png", cv2.IMREAD_UNCHANGED)
# seg.shape: (720, 1280), seg.dtype: uint8
```

## 클래스 설정

스크립트 상단의 `CLASS_MAP`과 `PALETTE`를 수정하여 클래스를 변경한다.

```python
CLASS_MAP = {
    "drivable_region": 1,
    "car": 2,
    "ego": 3,
    "fence": 4,
}
```
