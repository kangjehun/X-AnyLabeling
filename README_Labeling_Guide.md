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

## 라벨 클래스 정의

라벨링 JSON에는 다음 클래스 이름을 사용하고, 데이터 변환 시 아래 ID로 매핑한다.

| ID | Label | 설명 |
|---:|---|---|
| 0 | `background` | 배경 또는 미라벨링 영역 |
| 1 | `drivable_region` | 차량 주행 가능 영역 |
| 2 | `car` | 차량 |
| 3 | `ego` | 자차 노출 영역 |
| 4 | `fence` | 펜스 및 도로 방호물 |
| 5 | `curb` | 연석 |
| 6 | `roadsign` | 도로 표지판류 |
| 7 | `car_2dbbox` | 차량 2D bounding box (`rectangle`) |

X-AnyLabeling JSON에는 숫자 ID가 아니라 `label` 문자열과 `shape_type`이 저장되므로, 원본 라벨링 단계에서는 이름을 정확히 맞추는 것이 중요하다. 숫자 ID는 변환 단계에서 이름을 기준으로 다시 매핑할 수 있다. 다만 현재 semantic mask 생성 코드는 빈 mask를 값 `0`으로 초기화하므로, 현 파이프라인에서는 `background`를 ID 0으로 유지한다. 나머지 ID는 모든 후처리 및 학습 설정의 매핑을 함께 변경한다면 바꿀 수 있다.

Semantic segmentation은 `polygon`, 차량 2D bbox는 `car_2dbbox`와 `rectangle`을 사용한다. `car_2dbbox`는 semantic PNG mask 생성 대상에서 제외하고, 별도의 bbox 변환 단계에서 사용한다. Polygon 내부를 hole처럼 비울 때는 해당 영역에 `background` polygon을 덮어쓴다.

---

## 목차

- [환경 구성](#환경-구성)
  - [batch07 데이터 열기](#batch07-데이터-열기)
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
- GPU 사용 시: CUDA 11.x, 12.x 또는 13.x 및 호환 드라이버
- `curl` 또는 `wget` (`uv` 설치에 사용)

### 권장: uv 환경 생성 및 커스텀 소스 설치

`uv`는 Python이나 `pip`가 없어도 독립적으로 설치할 수 있다. 시스템
Python에 `pip`를 추가하지 않고, 저장소 안에 전용 가상환경을 만드는
방식을 권장한다.

```bash
# Conda 환경이 활성화되어 있다면 먼저 비활성화
conda deactivate  # Conda를 사용하지 않았다면 생략

# Linux/macOS/WSL2: uv 설치
curl -LsSf https://astral.sh/uv/install.sh | sh

# 현재 셸에 uv 경로 반영(또는 터미널을 다시 연다)
source "$HOME/.local/bin/env"
uv --version

# 저장소로 이동
cd /path/to/X-AnyLabeling

# 아래에서 사용하는 환경 하나만 생성하고 활성화한다.

# GPU, CUDA 12.x (기본 권장)
uv venv --python 3.12 .venv-cu12
source .venv-cu12/bin/activate
uv pip install -e ".[gpu]"

# GPU (CUDA 11.x)
uv venv --python 3.12 .venv-cu11
source .venv-cu11/bin/activate
uv pip install -e ".[gpu-cu11]"

# GPU (CUDA 13.x)
uv venv --python 3.12 .venv-cu13
source .venv-cu13/bin/activate
uv pip install -e ".[gpu-cu13]"

# CPU
uv venv --python 3.12 .venv-cpu
source .venv-cpu/bin/activate
uv pip install -e ".[cpu]"
```

한 번에 하나의 환경만 선택한다. 예를 들어 CUDA 12를 사용한다면
`.venv-cu12` 생성·활성화와 `.[gpu]` 설치 명령만 실행한다.

`-e`는 현재 저장소를 editable 모드로 설치한다. 이후 `custom/main`에서
Python 소스를 수정하면 패키지를 매번 다시 설치하지 않아도 변경 내용이
반영된다. 의존성이나 패키지 설정을 변경했을 때는 설치 명령을 다시
실행한다.

> `onnxruntime`과 `onnxruntime-gpu`가 한 환경에 동시에 설치되지 않도록
> 주의한다.

### 대안: Conda 환경 사용

Conda를 계속 사용하려면 환경을 만들 때 Python과 `pip`도 함께 설치한다.

```bash
conda create --name x-anylabeling python=3.12 pip -y
conda activate x-anylabeling

python -m pip install -U uv

cd /path/to/X-AnyLabeling

# CUDA 12.x 예시
uv pip install -e ".[gpu]"
```

프롬프트에 `(x-anylabeling)`이 표시되더라도 다음 명령에서 아무 경로도
나오지 않으면 Python 없이 빈 Conda 환경만 만들어진 상태다.

```bash
which python
```

이 경우 시스템 패키지인 `python-is-python3` 또는 `python3-pip`를 설치하지
말고 현재 Conda 환경에 Python을 추가한다.

```bash
conda install --name x-anylabeling python=3.12 pip
conda activate x-anylabeling

which python
python --version
```

`which python`은 Miniconda/Anaconda 아래의 `x-anylabeling/bin/python` 경로를,
`python --version`은 `Python 3.12.x`를 출력해야 한다.

### PyPI 배포판 설치

로컬 커스텀 코드를 사용하지 않고 공식 배포판만 설치할 때 사용한다.
이 프로젝트의 커스텀 변경을 사용하려면 이 명령 대신 위의
`uv pip install -e ".[gpu]"`와 같은 editable 설치를 사용한다.


```bash
# GPU (CUDA 12.x)
uv pip install "x-anylabeling-cvhub[gpu]"
```

### 설치 확인

```bash
which python
python --version
uv pip check
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
# uv 환경을 사용한 경우(CUDA 12 예시)
source .venv-cu12/bin/activate

# Conda 환경을 사용한 경우에는 위 명령 대신 다음을 실행
# conda activate x-anylabeling

# 기본 실행
xanylabeling

# 특정 디렉토리 열기
xanylabeling --filename /path/to/image_dir
```

모델은 `Ctrl+A`로 AI 모델 패널을 열어 선택한다. 최초 사용 시 `~/xanylabeling_data/models/` 경로에 자동 다운로드된다.

### batch07 데이터 열기

```bash
source /home/legatalee/Research/X-AnyLabeling/.venv-cu12/bin/activate

xanylabeling \
  --filename /home/legatalee/Dataset/INDY/PIDNET/batch07/images \
  --output /home/legatalee/Dataset/INDY/PIDNET/batch07/labels \
  --labels /home/legatalee/Dataset/INDY/PIDNET/batch07/labels/classes.txt \
  --validatelabel exact
```

- `--filename`: 이미지 디렉토리
- `--output`: JSON 라벨을 불러오고 저장할 디렉토리
- `--labels`: 클래스 목록 TXT 파일
- `--validatelabel exact`: 목록에 있는 클래스만 허용

다른 batch는 세 경로의 `batch07`을 해당 batch 이름으로 변경한다.

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
