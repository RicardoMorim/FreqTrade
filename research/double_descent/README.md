# Double Descent in Trading — Freqtrade/FreqAI Experiment

This module tests whether return prediction exhibits **double descent / benign overfitting** as Random Fourier Feature width (`P`) crosses and greatly exceeds the number of training observations (`N`).

It is a **replication + extension research experiment**, not a claim that the included trading strategy is profitable.

## Research questions

1. Does true out-of-sample prediction error worsen around `P / N ~= 1` and improve for `P >> N`?
2. Does prediction IC / R² show the same complexity curve?
3. Does any second descent survive Freqtrade fees and execution assumptions?
4. Is it robust across seeds, rolling windows, training-window sizes and another asset?
5. Do finite RFF models converge toward the exact RBF-kernel (`P -> infinity`) limit?

## Important design choices

### Prediction quality and trading quality are separate

The custom FreqAI model records true OOS metrics for every rolling prediction window **after inference**:

- MSE / MAE
- R²
- prediction-vs-realized correlation (IC)
- directional accuracy

The future target is used only for diagnostics after the prediction exists. It is never used by the strategy's entry/exit rules.

`run_freqtrade_sweep.py` aggregates the per-window sufficient statistics into `summary.csv`. Trading results remain Freqtrade's independent second layer.

### Hybrid primal / dual solver

For `P <= N`, the model uses an explicit `N x P` RFF matrix and solves in **primal** space. This avoids an unnecessary `N x N` solve for small models.

For `P > N`, RFF chunks are streamed into the **dual** `N x N` Gram matrix. The full `N x P` matrix is never materialized.

```text
P <= N                              P > N

X -> RFF Z (N x P)                 X -> RFF chunk
        |                                  |
        v                                  v
 solve P x P                         K += Zc @ Zc.T
        |                                  |
        v                                  v
     beta                              solve N x N
```

### RFF widths are genuinely nested

For fixed seed, chunk size, gamma and preprocessing, a smaller `P` uses an exact prefix of a larger `P` feature map. RNG is consumed in fixed-size blocks even for a partial final chunk. This avoids changing earlier RFF biases merely because `P` changed.

### Numerical policy

- `ridge_lambda > 0`: Cholesky solve with fallback.
- `ridge_lambda = 0`, `P <= N`: SVD pseudoinverse on `Z`, avoiding squared conditioning near interpolation.
- `ridge_lambda = 0`, `P > N`: Hermitian pseudoinverse in dual space.
- `float64` coefficients stay `float64`.
- default `matmul_precision=highest` prioritizes numerical integrity. Use `high` only for exploratory timing and confirm sensitive points again with `highest`.

## Files

- `rff_core.py` — standalone hybrid RFF / exact RBF-kernel implementation.
- `benchmark_rff.py` — measures fit/predict time and peak CUDA allocation locally.
- `synthetic_double_descent.py` — classical synthetic double-descent sanity check.
- `run_freqtrade_sweep.py` — resumable Freqtrade/FreqAI `P x seed` runner with OOS aggregation.
- `plot_sweep.py` — plots synthetic or market metrics against `P/N` with confidence bands.
- `freqtrade_assets/DoubleDescentRFFRegressor.py` — custom FreqAI model.
- `freqtrade_assets/DoubleDescentFreqaiStrategy.py` — deliberately simple trading layer.
- `freqtrade_assets/double_descent_freqai.example.json` — BTC/USDT futures 1h starting config.
- `tests/` — deterministic, nesting, hybrid-solver, precision and kernel-limit tests.

## Recommended execution order

### 1. Install / verify

From repository root:

```bash
python -m pip install -r research/double_descent/requirements.txt
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
python research/double_descent/install_into_user_data.py --force
```

### 2. Tests

```bash
cd research/double_descent
python -m pytest -q
cd ../..
```

All tests must pass before market runs.

### 3. Synthetic sanity check

```bash
python research/double_descent/synthetic_double_descent.py \
  --output research/double_descent/results/synthetic.csv

python research/double_descent/plot_sweep.py \
  research/double_descent/results/synthetic.csv \
  --metric test_mse \
  --output research/double_descent/results/synthetic_double_descent.png \
  --title "Synthetic double descent"
```

### 4. Download development data

The first market experiment uses BTC/USDT perpetual futures at 1h. Keep **2025 onward untouched** while choosing gamma, `P`, ridge and other methodology.

```bash
freqtrade download-data \
  --config user_data/configs/double_descent_freqai.example.json \
  --trading-mode futures \
  --timeframes 1h 5m \
  --timerange 20191001-20260801
```

### 5. Tiny FreqAI smoke run — discover actual N

```bash
python research/double_descent/run_freqtrade_sweep.py \
  --config user_data/configs/double_descent_freqai.example.json \
  --timerange 20230101-20230401 \
  --p-grid 1024 \
  --seeds 1 \
  --gamma 0.1 \
  --ridge 1e-6 \
  --train-period-days 60 \
  --backtest-period-days 30 \
  --experiment-id dd-v2-smoke
```

Inspect `results/freqtrade/summary.csv`. The fields `n_train_min`, `n_train_median`, `n_train_max`, `p_over_n_median`, OOS MSE/R²/IC and directional accuracy are generated automatically.

### 6. Benchmark the local GPU

Use the measured median `N` and input feature count. Example for `N ~= 1400`, `d ~= 30`:

```bash
python research/double_descent/benchmark_rff.py \
  --n-train 1400 \
  --n-predict 720 \
  --input-dim 30 \
  --p-grid 256,1024,1400,4096,16384,65536,100000 \
  --gamma 0.1 \
  --ridge 1e-6 \
  --device cuda \
  --matmul-precision highest
```

### 7. Gamma pilot — development data only

Do not tune gamma and complexity simultaneously. Pick roughly `P ~= 2N`, one seed, 30-day backtest windows and compare **OOS MSE/IC**, not trading profit.

Recommended gamma pilot:

```text
0.05, 0.10, 0.20, 0.50
```

Freeze gamma before the main complexity sweep.

### 8. Main exploratory complexity sweep

Build `P` from measured `N`, densely around interpolation and logarithmically away from it.

Recommended ratios:

```text
0.10, 0.25, 0.50, 0.75,
0.90, 0.95, 0.98, 1.00, 1.02, 1.05, 1.10,
1.25, 1.50, 2, 3, 5, 10, 25, 50
```

For `N = 1400`, approximately:

```text
140,350,700,1050,1260,1330,1372,1400,1428,1470,1540,1750,2100,2800,4200,7000,14000,35000,70000
```

Example:

```bash
python research/double_descent/run_freqtrade_sweep.py \
  --config user_data/configs/double_descent_freqai.example.json \
  --timerange 20200101-20250101 \
  --p-grid 140,350,700,1050,1260,1330,1372,1400,1428,1470,1540,1750,2100,2800,4200,7000,14000,35000,70000 \
  --seeds 1,2,3 \
  --gamma 0.10 \
  --ridge 1e-6 \
  --train-period-days 60 \
  --backtest-period-days 30 \
  --device cuda \
  --matmul-precision highest \
  --experiment-id dd-v2-main
```

If interrupted, rerun the same command with `--resume`. The runner skips successful runs. It fails fast by default after an error so an invalid setup does not waste an overnight run.

### 9. Plot prediction-space results first

```bash
python research/double_descent/plot_sweep.py \
  results/freqtrade/summary.csv \
  --metric oos_mse \
  --output results/freqtrade/oos_mse.png \
  --title "BTC 1h — OOS MSE vs P/N"

python research/double_descent/plot_sweep.py \
  results/freqtrade/summary.csv \
  --metric oos_ic \
  --output results/freqtrade/oos_ic.png \
  --title "BTC 1h — OOS IC vs P/N"
```

## After the first curve

Do not automatically run every expensive robustness test. If the first curve is interesting, continue with:

1. selected 100k / 250k / 500k / 1m finite-width spot checks;
2. exact RBF kernel `P -> infinity` reference;
3. ridgeless (`lambda=0`) sweep, with float64 checks near `P/N=1`;
4. restore 7-day retraining for selected widths;
5. 30 / 120-day training-window robustness;
6. noise-feature and shuffled-target null controls;
7. Freqtrade `lookahead-analysis` / `recursive-analysis`;
8. freeze methodology;
9. ETH external replication;
10. only then open the untouched 2025+ BTC holdout.

## Numerical cautions

- Do not interpret an interpolation spike until nearby points are repeated with `accumulator_dtype=float64` and `matmul_precision=highest`.
- `pinv_rtol` is an implicit rank threshold and must be reported for ridgeless results.
- `P` is nominal dimension; collect effective-rank diagnostics for selected widths.
- A million correlated dimensions are not equivalent to a million independent directions.

## Research integrity

- no random train/test split for time series;
- no final-holdout tuning;
- no hiding bad seeds;
- gamma frozen before the main `P` sweep;
- prediction metrics primary for the double-descent claim;
- trading metrics are a secondary economic validation;
- final conclusions include costs, drawdown and regime robustness.
