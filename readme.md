# Vicsek Model Simulator

**CPU/GPU 하이브리드** Vicsek 모델 시뮬레이터.

Metric(거리 기반)과 Topologic(k-NN 기반) 두 가지 상호작용 모델에 대해 **FOV × N × η × v** 4차원 파라미터 공간을 스캔하고 order parameter φ를 측정합니다. 입자 수(N)에 따라 CPU와 GPU를 **자동으로 전환**하여 최적의 성능을 발휘합니다.

<br>

## Features

- **하이브리드 라우팅** — N < 1,000은 CPU multiprocessing, N ≥ 1,000은 GPU CUDA 커널로 자동 전환
- **Cell List 알고리즘** — O(N²) → O(N) 스케일링. 대규모 시뮬레이션에서도 선형 복잡도
- **GPU 비동기 파이프라인** — 셀 리스트 구축을 포함한 전 과정을 GPU 내에서 완결 (PCI-e 병목 제거)
- **Numba fastmath SIMD** — AVX2/AVX-512 벡터화로 CPU 연산 20~30% 가속
- **수렴 자동 감지** — 슬라이딩 윈도우 기반 early stopping + 측정 윈도우 분리
- **중단 내성** — trial 단위 즉시 저장, 재시작 시 완료된 trial 자동 스킵
- **실시간 시각화** — matplotlib 기반 2-패널 애니메이션 (입자 분포 + φ 시계열)

<br>

## Installation

```bash
git clone https://github.com/<your-username>/vicsek-model.git
cd vicsek-model
```

### 필수 의존성

```bash
pip install numpy pandas
```

### 선택 의존성

```bash
# CPU 가속 (Numba JIT + fastmath SIMD 벡터화)
pip install numba

# GPU 모드 (CUDA 버전에 맞춰 설치)
pip install cupy-cuda12x      # CUDA 12.x
# pip install cupy-cuda11x    # CUDA 11.x

# 실시간 시각화
pip install matplotlib
sudo apt install python3-tk   # 또는 pip install PyQt5
```

<br>

## Quick Start

```bash
# 기본 실행: Topologic 모델, η 21개 자동 생성
python run.py --topologic --eta-auto

# 양쪽 모델 동시, FOV/N 범위 스캔
python run.py \
    --metric --topologic \
    --fov 60 120 180 240 300 360 \
    --N-range 40 3200 400 \
    --eta-auto

# 실시간 시각화 (GPU 필요)
python run.py --visualize \
    --viz-model topologic --viz-N 300 --viz-fov 360 --viz-eta 0.5
```

<br>

## Usage

### 파라미터 스캔

| 대상 | 옵션 | 예시 |
|---|---|---|
| **FOV** | `--fov` | `--fov 60 120 180 360` |
| **입자 수** | `--N` 또는 `--N-range` | `--N 40 400 800` 또는 `--N-range 40 3200 400` |
| **밀도** | `--density` 또는 `--density-range` | `--density 1.0 2.0 4.0` |
| **노이즈** | `--eta` 또는 `--eta-auto` | `--eta 0.0 0.25 0.5 1.0` |
| **속도** | `--v` 또는 `--v-range` | `--v 0.05 0.1 0.2` |

```bash
# 속도 스캔
python run.py --topologic --eta-auto --v-range 0.05 0.5 0.05

# 밀도로 N 자동 계산 (ρ = N/L²)
python run.py --topologic --eta-auto --L 25 --density 1.0 2.0 4.0
```

### 환경 제어

```bash
python run.py --force-cpu ...    # CPU 강제
python run.py --force-gpu ...    # GPU 강제
python run.py ...                # 자동 (N 크기별 최적 환경)
```

자동 모드에서의 동작:

```
N 값 범위 분석
  ├─ 모든 N < 1,000   → CPU 전용 (multiprocessing 병렬)
  ├─ 모든 N ≥ 1,000   → GPU 전용
  └─ N이 1,000을 걸침  → 하이브리드 (N별 동적 전환)
```

### 시각화 (GPU 전용)

```bash
# 실시간 창
python run.py --visualize \
    --viz-model topologic --viz-N 300 --viz-fov 360 --viz-eta 0.5

# 파일 저장
python run.py --visualize \
    --viz-model topologic --viz-N 300 --viz-eta 0.5 \
    --viz-save output.mp4    # .gif도 가능
```

### 백그라운드 실행

```bash
nohup python run.py \
    --metric --topologic \
    --fov 60 120 180 240 300 360 \
    --N-range 40 3200 400 \
    --eta-auto \
    --output-dir Full_Scan \
    > run.log 2>&1 &
```

<br>

## Noise Convention

η는 **[0, 1]로 정규화**되어 있습니다.

```
실제 노이즈 = η × uniform(-π, π)
```

| η | 의미 |
|---|---|
| 0.0 | 노이즈 없음 → 완전 정렬 (φ → 1) |
| 0.5 | 중간 노이즈 |
| 1.0 | 최대 노이즈 → 완전 무질서 (φ → 0) |

<br>

## Output Format

### 디렉토리 구조

```
Vicsek_Results/
├── 360_20.0_0.0500_0.3000_0.1000.csv
├── 360_20.0_0.0500_0.5000_0.1000.csv
├── 180_20.0_1.0000_0.7500_0.1000.csv
└── ...
```

파일명: **`{FOV}_{L}_{ρ}_{η}_{v}.csv`**

### CSV 구조

```csv
Time_Step,Metric_Trial_1,Metric_Trial_2,...,Topologic_Trial_1,...
0,0.847321,0.812456,...,0.923456,...
1,0.851234,0.815678,...,0.925678,...
...
```

- 행 수 = 측정 윈도우 (기본 300스텝)
- 수렴 후 별도 구간에서 측정한 정상 상태 φ 시계열
- 중단 후 재시작하면 완료된 trial/model은 자동 스킵

<br>

## Performance Optimizations

### 알고리즘

- **Cell List** — 인접 셀만 탐색하여 O(N) 복잡도 달성
- **argpartition** — k-NN 선택을 O(N log N) → O(N)으로 단축
- **링 버퍼 수렴 검사** — 고정 메모리로 슬라이딩 윈도우 비교
- **노이즈 청크** — 동적 생성으로 메모리 사용량 고정

### GPU

- **100% 비동기 파이프라인** — `backpropagate_cells` CUDA 커널로 CPU 왕복 제거
- **CUDA RawKernel** — max-heap k-NN, metric 이웃 탐색을 GPU 스레드에서 직접 수행
- **최소 동기화** — φ 수집 `.get()` 단 1회/스텝

### CPU

- **Numba `fastmath=True`** — AVX2/AVX-512 SIMD 벡터화 + 루프 언롤링 (20~30% 가속)
- **multiprocessing.Pool** — trial 단위 병렬화
- **`NUMBA_NUM_THREADS=1`** — Pool worker 내 Numba prange 충돌 방지

### 하이브리드

- **Per-N 동적 전환** — 소규모 N은 CPU (커널 오버헤드 회피), 대규모 N은 GPU (병렬 극대화)

<br>

## CLI Reference

<details>
<summary><b>전체 옵션 목록 (클릭하여 펼치기)</b></summary>

### 파라미터 설정

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--fov DEG [DEG ...]` | FOV 각도 (도) | 60 120 180 240 300 360 |
| `--N N [N ...]` | 입자 수 직접 지정 | 40 100 200 400 800 1200 1600 |
| `--N-range START STOP STEP` | 입자 수 범위 | — |
| `--density RHO [RHO ...]` | 밀도 ρ=N/L² | — |
| `--density-range START STOP STEP` | 밀도 범위 | — |
| `--L FLOAT` | 시스템 크기 | 20.0 |
| `--eta ETA [ETA ...]` | 노이즈 (0~1) | 0.0 ~ 1.0 (17개) |
| `--eta-auto` | η = 0.0, 0.05, ..., 1.0 (21개) | — |
| `--v V [V ...]` | 입자 속도 | 0.1 |
| `--v-range START STOP STEP` | 속도 범위 | — |

### 모델 & 환경

| 옵션 | 설명 |
|---|---|
| `--metric` | Metric 모델 실행 |
| `--topologic` | Topologic 모델 실행 (기본값) |
| `--force-cpu` | CPU 강제 |
| `--force-gpu` | GPU 강제 |

### 실행 파라미터

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--trials N` | trial 수 | 100 |
| `--max-steps N` | 최대 스텝 (안전장치) | 10,000 |
| `--chunk N` | noise 청크 크기 | 1,000 |
| `--workers N` | CPU worker 수 | CPU 코어 수 |
| `--output-dir DIR` | 출력 디렉토리 | Vicsek_Results |

### 시각화 (GPU 전용)

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--visualize` | 시각화 모드 활성화 | — |
| `--viz-model {metric,topologic}` | 시각화 모델 | topologic |
| `--viz-N INT` | 입자 수 | 200 |
| `--viz-density RHO` | 밀도 → N 자동 | — |
| `--viz-L FLOAT` | 시스템 크기 | 20.0 |
| `--viz-fov INT` | FOV (도) | 360 |
| `--viz-eta FLOAT` | 노이즈 (0~1) | 0.5 |
| `--viz-frames INT` | 최대 프레임 (0=무제한) | 2,000 |
| `--viz-interval INT` | 프레임 간격 (ms) | 30 |
| `--viz-save FILE` | .mp4 또는 .gif 저장 | — |

</details>

<br>

## Requirements

| 패키지 | 용도 | 필수 여부 |
|---|---|---|
| Python 3.10+ | 런타임 | **필수** |
| numpy | 배열 연산 | **필수** |
| pandas | CSV 관리 | **필수** |
| numba | CPU JIT 가속 | 선택 (미설치 시 NumPy 폴백) |
| cupy | GPU 연산 | 선택 (미설치 시 CPU 전용) |
| matplotlib | 시각화 | 선택 (`--visualize` 시 필요) |

<br>

## License

Selian