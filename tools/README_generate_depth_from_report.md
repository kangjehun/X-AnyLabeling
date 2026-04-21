# generate_depth_from_report.py

`segmentation_only_items.txt`에 있는 항목만 골라 상대 깊이 라벨 `*_depth.npy`를 생성하는 배치 스크립트.

- 입력: `label_check_reports/segmentation_only_items.txt`
- 대상: report 표의 `Scenario | Camera | Timestamp | JSON Path`
- 출력: 각 카메라 폴더 아래 `x-anylabeling-depth/{timestamp}_depth.npy`

## 용도

기존 수작업:

1. 시나리오 디렉토리를 `xanylabeling --filename ...`으로 열기
2. `Depth Anything V2 (ViT-Large)` 선택
3. `segmentation_only_items.txt`에 적힌 파일만 하나씩 실행

이 과정을 report 기반 배치 실행으로 대체한다.

## 실행

전체 대상 실행:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt"
```

실행 전 확인만:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt" \
  --dry-run
```

기존 파일까지 다시 생성:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt" \
  --overwrite
```

## 자주 쓰는 옵션

특정 시나리오만:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt" \
  --scenario a1_scenario_16
```

특정 카메라만:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt" \
  --camera CAMERA_1
```

일부만 테스트:

```bash
python tools/generate_depth_from_report.py \
  --report "/media/kangjehun/T7 Shield/sample data/MORAI_SAMPLE/label_check_reports/segmentation_only_items.txt" \
  --limit 10
```

## 옵션 요약

| 옵션 | 설명 |
|------|------|
| `--report` | `segmentation_only_items.txt` 경로 |
| `--dry-run` | 실제 생성 없이 대상만 확인 |
| `--overwrite` | 이미 존재하는 `_depth.npy`도 다시 생성 |
| `--scenario` | 특정 시나리오만 처리, 여러 번 지정 가능 |
| `--camera` | 특정 카메라만 처리, 여러 번 지정 가능 |
| `--limit` | 앞에서부터 N개만 처리 |
| `--config` | depth 모델 yaml 경로. 기본값은 `depth_anything_v2_vit_l.yaml` |
| `--work-dir` | X-AnyLabeling 설정/모델 캐시 기준 경로. 기본값은 홈 디렉토리 |

## 동작 방식

- report 표에서 `.json` 경로를 읽는다.
- 같은 위치의 이미지 파일(`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`)을 찾는다.
- `Depth Anything V2 (ViT-Large)`를 직접 호출해 depth를 생성한다.
- 출력 `.npy`가 `float32`, 2D, 범위 `[0, 1]`인지 검증한다.
- 기본적으로 이미 존재하는 `_depth.npy`는 건너뛴다.

## 출력 예시

```text
[1/3] a1_scenario_16 CAMERA_1 0 -> ok: shape=(720, 1280), range=[0.0000, 1.0000]
```

## 참고

- 이 스크립트는 현재 리포지토리의 수정된 depth 저장 로직을 전제로 한다.
- 모델 config 기본값:
  `anylabeling/configs/auto_labeling/depth_anything_v2_vit_l.yaml`
- 기본 work dir:
  `/home/kangjehun`
