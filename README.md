# More than a Hack 2027 — Phase 1: Ball Possession
## Official Submission by Team Helsion

**Competition:** [More than a Hack 2027](https://github.com/metrica-sports/more-than-a-hack-2027)  
**Organised by:** [Metrica Sports](https://metrica-sports.com), [FC Barcelona (Barça Innovation Hub)](https://www.fcbarcelona.com), and [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)  
**Team Name:** **Helsion**  
**Submission Deliverable:** [`Helsion.csv`](Helsion.csv) | **Technical Report:** [`Helsion.md`](Helsion.md)

---

## Benchmark Results (Public Ground Truth)

Evaluated with the official competition scorer (`python evaluation/run_evaluation.py`):

| Submission | `macroF1` (Per-frame) | `spellF1` (Official Ranking Key) | Possessions (GT: 17) | Mean Duration (GT: 14.2s) | Dead Ball Recall | Dead Ball F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0 s | 33.6% | 0.4197 |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6 s | 28.2% | 0.3857 |
| `baseline_smooth50` *(Floor)* | 0.6207 | 0.3256 | 26 (+9) | 10.8 s | 21.4% | 0.3225 |
| `baseline_smooth125`* | 0.5812 | 0.4242 | 16 (-1) | 18.5 s | 7.9% | 0.1464 |
| **`Helsion v2.0` (Ours)** | **0.8065** | **0.4235** | **17 (Exact)** | **13.1 s** | **91.5%** | **0.7922** |

*\*`baseline_smooth125` achieved 0.42 on the public 5 minutes by predicting almost zero dead balls (7.9% recall), which collapses to 0.2242 on the full match. `Helsion` achieves balanced performance across all three classes ($H: 0.8135$, $A: 0.8138$, $D: 0.7922$) and exactly reconstructs the 17 possessions.*

---

## Core Innovations in `pipeline.py`

1. **Ball Kinematic Trajectory Reconstruction:**
   * Solves the **40.7% missing ball problem** by fitting continuous kinematic interpolation (up to 150 frames = 6 seconds) between visible ball coordinates. Passes and crosses maintain continuous in-flight tracking.
2. **Time-to-Intercept (TTI) Arrival Physics:**
   * Replaces static Euclidean distance with velocity-adjusted arrival times:
     $$\tau_i = \frac{\|\mathbf{p}_i - \mathbf{b}\|}{\max\left(2.5, \min(s_i + 3.0, 9.0)\right)}$$
   * Sprinters accelerating towards the ball are correctly awarded possession over stationary defenders.
3. **Collective Tactical Morphology & Whistle Detection:**
   * Detects referee stoppages via collective player deceleration ($\bar{s}_{\text{players}} < 1.32\text{ m/s}$), out-of-bounds spatial coordinates ($x, y \notin [-0.01, 1.01]$), and Smart Tagging set-piece preparation windows.
   * Regularized via morphological closing (50 frames) and opening (80 frames) to eliminate micro-stoppages.
4. **Possession Inertia & Smart Tagging Bayesian Priors:**
   * Enforces possession continuity during pass flights and duels.
   * Uses 237 Smart Tagging offensive phases (`Build Up`, `Progression`, `Creating Chances`) as Bayesian priors.
   * Applies a 35-frame debounced hysteresis turnover window to eliminate transition jitter.

---

## Quickstart & Reproduction

To reproduce the exact submission file `Helsion.csv` from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the pipeline (generates Helsion.csv in ~15 seconds)
python pipeline.py --output Helsion.csv

# 3. Validate formatting compliance
python evaluation/validate_submission.py Helsion.csv

# 4. Run official evaluation against baselines
python evaluation/run_evaluation.py
```

---

## Repository Structure

```
├── pipeline.py                 # Core pipeline regenerating Helsion.csv
├── Helsion.csv                 # Official submission file (199,319 rows)
├── Helsion.md                  # Comprehensive technical documentation report
├── requirements.txt            # Pinned dependencies (numpy, pandas, scipy)
├── games.py                    # Match constants, pitch dimensions, and video frame alignments
├── evaluation/
│   ├── run_evaluation.py       # Official competition scorer
│   ├── validate_submission.py  # Submission format validator
│   ├── possession_metrics.py   # Metric calculation library
│   ├── gt/                     # Public 5-minute ground truth
│   └── results/                # Scored submission files
├── starter/                    # Reference baseline and loader utilities
└── data/                       # Gated match assets (gitignored)
```

---

## Technical Report

For full methodology, asset ablation, domain rules, and error analyses, refer to the [Technical Report (`Helsion.md`)](Helsion.md).
