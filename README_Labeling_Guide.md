# X-AnyLabeling 활용 가이드

## 원본 소스 수정 사항

본 프로젝트에서 X-AnyLabeling 원본 코드를 다음과 같이 수정하였다. upstream 업데이트 시 충돌 여부를 확인할 것.

### 수정 1. Depth 저장 경로 변경

`anylabeling/services/auto_labeling/depth_anything_v2.py:132`

```python
# 변경 전 (원본): 이미지 폴더의 상위에 저장 → 카메라 간 파일 충돌
save_path = os.path.join(image_dir_path, "..", self.save_dir)

# 변경 후: 이미지 폴더 하위에 저장 → 카메라별 독립 저장
save_path = os.path.join(image_dir_path, self.save_dir)
```

출력 경로가 `CAMERA_X/../x-anylabeling-depth/`에서 `CAMERA_X/x-anylabeling-depth/`로 변경됨.

### 수정 2. Depth 저장 시 NPY 단독 저장

`anylabeling/services/auto_labeling/depth_anything_v2.py:139-148`

```python
# 변경 전 (원본): save_raw_depth=true 시 PNG + NPY 둘 다 저장
cv2.imwrite(save_file, depth_visual)
if self.save_raw_depth:
    np.save(depth_raw_file, depth_calibrated)

# 변경 후: save_raw_depth=true 시 NPY만, false 시 PNG만 저장
if self.save_raw_depth:
    np.save(depth_raw_file, depth_calibrated)
else:
    cv2.imwrite(save_file, depth_visual)
```

### 수정 3. Depth Anything V2 ViT-L Config 변경

`anylabeling/configs/auto_labeling/depth_anything_v2_vit_l.yaml`

```yaml
# 변경 전 (원본)
render_mode: color  # 'color' or 'gray'

# 변경 후: float32 .npy 저장 활성화
render_mode: gray
min_depth: 0.0
max_depth: 1.0
save_raw_depth: true
```

`min_depth: 0.0, max_depth: 1.0`이면 내부 계산 `normalized * (1.0 - 0.0) + 0.0`이 항등 변환이 되어 relative depth (float32, [0, 1])가 그대로 `.npy`로 저장됨.

---

## 목차

- [환경 구성](#환경-구성)
- [Part A. Semantic Segmentation](#part-a-semantic-segmentation)
  - [A1. 개요](#a1-개요)
  - [A2. 지원 모델](#a2-지원-모델)
  - [A3. 자동 Annotation 워크플로우](#a3-자동-annotation-워크플로우)
  - [A4. Label 수정 및 재가공](#a4-label-수정-및-재가공)
  - [A5. 8-bit Encoded PNG Mask Export](#a5-8-bit-encoded-png-mask-export)
  - [A6. Mask Import](#a6-mask-import)
  - [A7. Custom 모델 학습 후 재활용](#a7-custom-모델-학습-후-재활용)
- [Part B. Depth Estimation](#part-b-depth-estimation)
  - [B1. 개요](#b1-개요)
  - [B2. 지원 모델](#b2-지원-모델)
  - [B3. Relative Depth Label 생성](#b3-relative-depth-label-생성)
  - [B4. 출력 결과](#b4-출력-결과)
  - [B5. 학습 데이터로 활용](#b5-학습-데이터로-활용)
- [공통 참고사항](#공통-참고사항)
  - [단축키 요약](#단축키-요약)
  - [기타 편의 기능](#기타-편의-기능)
  - [참고 자료](#참고-자료)

---

## 환경 구성

### 사전 요구사항

- Python 3.11 ~ 3.13 (3.12 권장)
- GPU 사용 시: CUDA 11.x 또는 CUDA 12.x 및 호환 드라이버
- Conda (Miniconda 또는 Anaconda)

### Conda 환경 생성 및 설치

```bash
# 환경 생성
conda create --name x-anylabeling python=3.12 -y
conda activate x-anylabeling

# 소스 기반 설치
cd /path/to/X-AnyLabeling
pip install -U uv

# GPU (CUDA 12.x)
uv pip install -e ".[gpu]"

# GPU (CUDA 11.x)
# uv pip install -e ".[gpu-cu11]"

# CPU
# uv pip install -e ".[cpu]"
```

> `onnxruntime`과 `onnxruntime-gpu`가 동시에 설치되지 않도록 주의한다.

pip 패키지 설치:

```bash
# GPU (CUDA 12.x)
uv pip install x-anylabeling-cvhub[gpu]
```

### 설치 확인

```bash
xanylabeling checks
```

출력 예시:

```
Application
────────────────────────────────────────────────────────────
  App Name:          X-AnyLabeling
  App Version:       4.0.0-beta.3
  Preferred Device:  GPU
────────────────────────────────────────────────────────────
System
────────────────────────────────────────────────────────────
  GPU:               CUDA:0 (NVIDIA GeForce RTX 2080 Super, 8192MiB)
  CUDA:              V12.4.99
  Python Version:    3.12.13
────────────────────────────────────────────────────────────
Packages
────────────────────────────────────────────────────────────
  ONNX Runtime GPU Version:                1.24.4
  OpenCV Contrib Python Headless Version:  4.13.0.92
────────────────────────────────────────────────────────────
```

### 실행

```bash
conda activate x-anylabeling

# 기본 실행
xanylabeling

# 특정 디렉토리 열기
xanylabeling --filename /path/to/image_dir
```

모델은 `Ctrl+A`로 AI 모델 패널을 열어 선택한다. 최초 사용 시 `~/xanylabeling_data/models/` 경로에 자동 다운로드된다.

---

## Label 수정 및 재가공

### Polygon 편집

| 작업 | 조작 |
|------|------|
| Polygon 편집 모드 | `Ctrl+J` |
| Vertex 추가 | polygon edge 클릭 후 드래그 |
| Vertex 삭제 | `Shift+Click` on vertex |
| Shape 이동 | shape 내부 드래그 |
| Label 변경 | shape 더블 클릭 |

### 일괄 편집

| 기능 | 접근 |
|------|------|
| Shape Manager | `Alt+S` 또는 Tools > Shape Manager |
| Label Manager | Tools > Label Manager (클래스명 일괄 변경) |
| Group ID Manager | Tools > Group ID Manager |
| Shape Type Conversion | Tools > Shape Converter |

## 단축키 요약

| 단축키 | 기능 |
|--------|------|
| `Ctrl+A` | AI 모델 선택 패널 열기 |
| `i` | 현재 이미지에 대해 AI 추론 실행 |
| `Ctrl+M` | 모든 이미지에 대해 일괄 추론 |
| `Ctrl+I` | 이미지 파일 열기 |
| `Ctrl+O` | 비디오 파일 열기 |
| `Ctrl+J` | Polygon 편집 모드 |
| `Ctrl+S` | 저장 |
| `Q` | Positive point prompt |
| `E` | Negative point prompt |
| `D` | 다음 이미지 |
| `A` | 이전 이미지 |
| `Ctrl+Z` | 실행 취소 |
| `Alt+S` | Shape Manager |
