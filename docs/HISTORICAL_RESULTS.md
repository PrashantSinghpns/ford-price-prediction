# Historical evaluation record

The text below preserves the prior README result for traceability. Its source CSV and saved training artifacts are not included here, so these numbers were not reproduced in this review.

The old regression workflow selected its winner using test RMSE. Those reported scores are selection-biased and do not validate the revised training-only cross-validation workflow.

## Results

On the supplied 17,966-row CSV, one invalid future-year record was removed. With an 80/20 split (`random_state=42`), the selected Random Forest achieved **MAE £837.36**, **RMSE £1,218.05**, and **R² 0.9336** on 3,593 held-out rows. These are one-split estimates, not a guarantee of real market pricing performance.
