## Known Simulation Design Decisions

**Heavy tier imbalance (27 control vs 36 treatment):**
The heavy spender tier (3% of users, n=63) shows a mild 
assignment imbalance due to small sample size and hash-based 
randomisation. This is intentional — it creates a realistic 
scenario for the SRM detection layer (Phase 3) to flag and 
investigate. In production, this tier would have thousands 
of users and hashing would produce near-perfect splits.