# Double Descent in Trading — Freqtrade/FreqAI Experiment

This research module tests whether return prediction exhibits **double descent / benign overfitting** as the number of Random Fourier Features (`P`) crosses and greatly exceeds the number of training observations (`N`).

The module is designed as a **replication + extension experiment**, not as a claim that the included strategy is profitable.

## Research questions

1. Does out-of-sample prediction error deteriorate near `P / N ~= 1` and improve again for `P >> N`?
2. Does the same pattern appear in Freqtrade trading metrics after fees and realistic execution assumptions?
3. Is any second descent robust across random seeds and rolling windows?
4. Does finite-width RFF performance converge toward the exact RBF-kernel (`P -> infinity`) limit?
5. Does the effect survive longer training windows, which reduces the chance that the model is merely acting like a recency/momentum rule?

## Why this lives under `research/`

Freqtrade intentionally ignores most files under `user_data`. The canonical experiment therefore stays in this tracked directory. Run the installer to copy the FreqAI model, strategy and example config into your local `user_data` directory.

## Components

- `rff_core.py` — streamed finite-width Random Fourier Feature ridge/minimum-norm regression plus exact RBF-kernel limit.
- `synthetic_double_descent.py` — sanity check on synthetic data before touching markets.
- `plot_sweep.py` — plots metric vs `P/N` on a logarithmic x-axis.
- `run_freqtrade_sweep.py` — launches repeatable Freqtrade/FreqAI backtests over a `P x seed` grid.
- `freqtrade_assets/DoubleDescentRFFRegressor.py` — custom FreqAI model.
- `freqtrade_assets/DoubleDescentFreqaiStrategy.py` — intentionally simple long/short strategy.
- `freqtrade_assets/double_descent_freqai.example.json` — starting configuration for BTC/USDT perpetual 1h experiments.
- `tests/` — deterministic RFF and kernel-limit checks.

## Important implementation property

The custom model does **not** create one million pandas columns.

For finite RFF width `P`, it generates feature chunks on GPU and accumulates the `N x N` Gram matrix:

```text
base market features (N x d)
        |
        v
RFF chunk (N x chunk_size)
        |
        v
K += Z_chunk @ Z_chunk.T
        |
        v
next RFF chunk
```

Memory therefore scales primarily with `N^2` plus one feature chunk, rather than `N x P`.

The exact implementation cost is `O(N^2 P)`. For very large `N`, use the RBF-kernel-limit mode as the `P -> infinity` reference and reserve explicit 250k/500k/1m-feature runs for selected windows.

## 1. Install Freqtrade and GPU dependencies

Follow Freqtrade's normal installation instructions with FreqAI enabled. PyTorch is required by this experiment. On an NVIDIA machine install the CUDA-enabled PyTorch build matching your driver/toolkit rather than a CPU-only wheel.

From the repository root, install plotting/test extras for this research module:

```bash
python -m pip install -r research/double_descent/requirements.txt
```

Verify CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 2. Install the Freqtrade assets

From the Freqtrade repository root:

```bash
python research/double_descent/install_into_user_data.py
```

Use `--force` to replace a previous installation:

```bash
python research/double_descent/install_into_user_data.py --force
```

## 3. Synthetic sanity check first

Before using market data, verify that the experiment harness can recover the classical interpolation-region behavior:

```bash
cd research/double_descent
python synthetic_double_descent.py --output results/synthetic.csv
python plot_sweep.py results/synthetic.csv \
  --metric test_mse \
  --output results/synthetic_double_descent.png \
  --title "Synthetic double descent"
```

The exact curve varies with seed/noise, but the run should show training error collapsing as model dimension reaches/exceeds the sample size and a pronounced generalization penalty around the interpolation region under suitable noise.

## 4. Download market data

The example uses Binance futures, BTC/USDT perpetual, 1h candles. Choose a timerange that is available from your exchange/data source.

Example:

```bash
freqtrade download-data \
  --config user_data/configs/double_descent_freqai.example.json \
  --trading-mode futures \
  --timeframes 1h 5m \
  --timerange 20190101-20260101
```

The 5m data is optional but recommended for `--timeframe-detail 5m` execution validation.

## 5. Smoke test one FreqAI model

Start small:

```bash
freqtrade backtesting \
  --config user_data/configs/double_descent_freqai.example.json \
  --strategy DoubleDescentFreqaiStrategy \
  --freqaimodel DoubleDescentRFFRegressor \
  --timerange 20230101-20240101
```

The example strategy uses only a compact, causal market-state feature vector. The huge dimensionality is created *inside the model*, not by duplicating technical indicators in the dataframe.

## 6. Complexity sweep

Run a first finite-width sweep:

```bash
python research/double_descent/run_freqtrade_sweep.py \
  --config user_data/configs/double_descent_freqai.example.json \
  --timerange 20200101-20250101 \
  --p-grid 64,128,256,512,1024,2048,4096,8192,16384,32768,65536 \
  --seeds 1,2,3,4,5 \
  --device cuda \
  --timeframe-detail 5m
```

Do **not** start with one million features. First establish whether the curve is interesting around `P/N ~= 1` and through moderate overparameterization. Only then add selected points such as 100k, 250k, 500k and 1m.

## 7. Kernel-limit run (`P -> infinity`)

Copy the example config and change:

```json
"model_training_parameters": {
  "mode": "rbf_kernel_limit",
  "gamma": 0.5,
  "ridge_lambda": 0.000001,
  "device": "cuda"
}
```

This evaluates the exact RBF kernel corresponding to the infinite-width limit of the Random Fourier Feature map without explicitly generating millions of features.

## 8. Training-window experiment

The initial example uses 60 training days. Repeat the sweep with at least:

- 15 days
- 30 days
- 60 days
- 120 days
- 365 days

The important variable is `P / N`, not `P` alone. Each model fit writes `double_descent_diagnostics.jsonl` inside its FreqAI model-data directory, including the actual training-row count and `P/N`.

## 9. Mandatory validation

For candidate strategies run Freqtrade's built-in bias/indicator checks in addition to ordinary backtesting:

```bash
freqtrade lookahead-analysis \
  --config user_data/configs/double_descent_freqai.example.json \
  --strategy DoubleDescentFreqaiStrategy \
  --freqaimodel DoubleDescentRFFRegressor \
  --timerange 20230101-20240101
```

and:

```bash
freqtrade recursive-analysis \
  --config user_data/configs/double_descent_freqai.example.json \
  --strategy DoubleDescentFreqaiStrategy \
  --timerange 20230101-20240101
```

A result that fails leakage/bias checks is invalid regardless of Sharpe or profit.

## 10. Experimental phases

### Phase A — synthetic

Validate the harness and interpolation behavior.

### Phase B — finite RFF on BTC 1h

Sweep `P/N` densely around 1 and then logarithmically into the overparameterized regime. Use at least 5 seeds initially and 20 for final results.

### Phase C — RBF kernel limit

Compare finite RFF widths with the `P -> infinity` reference.

### Phase D — training window robustness

Repeat across multiple `train_period_days` values.

### Phase E — external replication

Freeze the method and repeat on ETH/USDT. Do not tune the method on ETH after seeing the BTC result.

### Phase F — untouched final holdout

Keep the most recent interval fully untouched until feature definitions, gamma selection, ridge handling, signal rules, costs and model-selection procedure are frozen.

## Metrics

Track prediction metrics separately from trading metrics.

Prediction layer:

- out-of-sample MSE
- out-of-sample R2
- prediction/realized-return correlation (IC)
- directional accuracy

Trading layer (Freqtrade):

- net return
- Sharpe
- Sortino
- max drawdown
- profit factor
- turnover / trade count
- long vs short performance

The primary scientific plot is metric vs `log10(P/N)`, with confidence intervals across seeds and a vertical marker at `P/N = 1`.

## Numerical cautions

- Do not use FP16 for the kernel solve near the interpolation threshold.
- `ridge_lambda=0` uses a pseudoinverse and therefore depends on `pinv_rtol`; report that value in final results.
- A numerical singularity can look like an interpolation spike. Repeat sensitive points in `float64` before interpreting them scientifically.
- RFF dimensions are nested only when seed, chunk size, gamma and input preprocessing are held constant.

## Research integrity

Do not select the best `P` from the final holdout. Do not hide seeds that perform poorly. Report all planned widths/seeds and distinguish exploratory plots from the frozen final test.

The included strategy is intentionally not optimized for profit. Its job is to expose changes in predictive-model quality to a mature backtesting engine with as little strategy-level confounding as possible.
