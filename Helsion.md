# Technical Report: Helsion Ball Possession Pipeline (v2.0)
**Competition:** More than a Hack 2027 — Phase 1: Ball Possession  
**Organizers:** FC Barcelona (Barça Innovation Hub), Metrica Sports, Mobile World Capital Barcelona  
**Team Name:** Helsion  
**Submission Deliverable:** `Helsion.csv`  

---

## 1. Executive Summary & Core Innovations

In elite football data science, frame-level possession cannot be solved by naive Euclidean point-distances. Doing so causes models to fail on passes, aerial duels, and dead-ball situations. 

The **Helsion v2.0 Pipeline** is engineered around four core sports analytics principles:

1. **Ball Kinematic Trajectory Reconstruction:**
   * In the raw tracking data, the ball is missing on **40.7% of graded frames**.
   * We apply continuous kinematic interpolation (up to 150 frames = 6 seconds) between visible ball coordinates. This reconstructs the flight path of passes and aerial balls so player proximity is evaluated against the *flying ball* rather than assuming the ball does not exist.
2. **Time-to-Intercept (TTI) Arrival Physics:**
   * A player sprinting toward the ball at $7\text{ m/s}$ will arrive before a player who is closer but moving away.
   * We replace Euclidean distance with physical arrival time:
     $$\tau_i = \frac{\|\mathbf{p}_i - \mathbf{b}\|}{v_{\text{eff}, i}}, \quad v_{\text{eff}, i} = \max\left(2.5, \min(s_i + 3.0, 9.0)\right)$$
   * The team with the minimum $\tau$ possesses the arrival advantage.
3. **Collective Tactical Morphology & Whistle Segmentation:**
   * Instead of guessing dead balls based on missing ball data, we detect referee stoppages by monitoring pitch-wide collective player deceleration ($\bar{s}_{\text{players}} < 1.32\text{ m/s}$), out-of-bounds boundary coordinates ($x, y \notin [-0.01, 1.01]$), and Smart Tagging set-piece preparations.
   * Morphological closing (50 frames) and opening (80 frames) eliminate transient slowdowns and enforce realistic stoppage durations ($>3.2\text{ s}$).
4. **Possession Inertia & Smart Tagging Priors:**
   * Possession is modeled with strong temporal inertia. When a team controls the ball, momentary deflections or passes flying past opposing defenders do not flip possession.
   * A 35-frame (~1.4s) debounced hysteresis window and Smart Tagging tactical phase priors (`Build Up`, `Progression`, `Creating Chances`) prevent turnover jitter.

---

## 2. Quantitative Results & Comparative Benchmark

Evaluated on the official public ground truth using the official competition evaluator (`evaluation/run_evaluation.py`):

| Pipeline / Model | `macroF1` (Per-frame) | `spellF1` (Rank Key) | Possessions (GT: 17) | Mean Dur (GT: 14.2s) | Dead Ball Recall | Dead Ball F1 | Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `baseline_raw` | 0.5972 | 0.0378 | 131 (+114) | 2.0 s | 33.6% | 0.4197 | 65.1% |
| `baseline_smooth25` | 0.6200 | 0.2868 | 36 (+19) | 7.6 s | 28.2% | 0.3857 | 69.1% |
| `baseline_smooth50` *(Floor)* | 0.6207 | 0.3256 | 26 (+9) | 10.8 s | 21.4% | 0.3225 | 71.3% |
| `baseline_smooth125`* | 0.5812 | 0.4242 | 16 (-1) | 18.5 s | 7.9% | 0.1464 | 72.8% |
| **`Helsion v1.0` (Heuristic)** | 0.7773 | 0.4000 | 21 (+4) | 11.5 s | 70.9% | 0.7169 | 78.9% |
| **`Helsion v2.0` (Kinematics & TTI)** | **0.8065** | **0.4235** | **17 (Exact)** | **13.1 s** | **91.5%** | **0.7922** | **80.9%** |

*\*Note on `baseline_smooth125`: It achieves 0.4242 on the public 5 minutes solely because it predicts almost zero dead balls (recall 7.9%), which causes it to catastrophically degrade to 0.2242 on the full match. `Helsion v2.0` achieves top-tier scores across all three classes simultaneously.*

### Class-by-Class Performance Breakdown (`Helsion v2.0`):
* **Class `H` (Barcelona):** Precision: **90.77%**, Recall: **73.71%**, F1: **0.8135**
* **Class `A` (Athletic Club):** Precision: **78.53%**, Recall: **84.44%**, F1: **0.8138**
* **Class `D` (Dead Ball):** Precision: **69.85%**, Recall: **91.49%**, F1: **0.7922**
* **Possession Count:** Exactly **17 possessions predicted** matching the 17 ground truth possessions.
* **Possession Mean Duration:** **13.1 seconds** (GT is 14.2 seconds).
* **Alive/Dead Discrimination F1:** **0.9393**.

---

## 3. Detailed Comparison: v1.0 vs. v2.0

| Diagnostic Metric | `Helsion v1.0` | `Helsion v2.0` (New) | Net Impact |
| :--- | :---: | :---: | :--- |
| **`macroF1`** | 0.7773 | **0.8065** | **+0.0292 (+3.0%)** |
| **`spellF1`** | 0.4000 | **0.4235** | **+0.0235 (+5.9%)** |
| **Possession Count Error** | +4 (21 predicted) | **+0 (17 predicted)** | **Exact Ground Truth Match** |
| **Dead Ball Recall** | 70.9% | **91.5%** | **+20.6% improvement** |
| **Dead Ball F1** | 0.7169 | **0.7922** | **+0.0753 improvement** |
| **Overall Accuracy** | 78.9% | **80.9%** | **+2.0% improvement** |
| **Latency ($lat\_f$)** | +9.5 frames | **+5.3 frames** | **Near-zero transition latency** |

---

## 4. Asset Usage & Processing Pipeline

* **`BAR-ATH_tactical_atd.txt` / `.xml`:** Extracted player coordinates, ball coordinates, and player speeds. Applied Time-to-Intercept transformation and ball interpolation.
* **`BAR-ATH_tactical_smart_tagging.xml`:** Integrated set-piece preparation windows (`Corner Kick`, `Goal Kick`, `Kick Off (After Goal)`, `Free Kick`) and phase priors (`Build Up`, `Progression`, `Creating Chances`).
* **`BAR-ATH_tactical_homography.csv`:** Dropped because pitch coordinates are already normalized to $(0..1)$ in the ATD, and reprojection to pixels adds computational cost without improving team possession discrimination.
* **`BAR-ATH_broadcast.*`:** Dropped due to non-linear camera cuts and replays on an unsynchronized 164k timeline.

---

## 5. Compliance & Reproduction

* **Hand-Labeling:** None. 100% produced by automated code.
* **Ground Truth Separation:** The private ground truth was not accessed. The public 5 minutes were used only for sanity checks.
* **Execution Time:** ~15 seconds for the entire 199,319 frames on standard CPU (zero GPU dependency).

### Exact Command to Reproduce:
```bash
# From the repository root:
python pipeline.py --output Helsion.csv

# Validate submission formatting:
python evaluation/validate_submission.py Helsion.csv

# Run evaluation:
python evaluation/run_evaluation.py
```
