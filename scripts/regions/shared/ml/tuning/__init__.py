"""
Shared hyperparameter-tuning package for South America and USA models.

Why needed:
- Centralizes reusable tuning logic so region scripts stay thin and consistent.

Input:
- Runtime configuration from region entry scripts and environment variables.

Output:
- Reusable functions that produce tuned-parameter artifacts and tuning logs.
"""
