## Known Simulation Design Decisions

**Heavy tier imbalance (27 control vs 36 treatment):**
The heavy spender tier (3% of users, n=63) shows a mild 
assignment imbalance due to small sample size and hash-based 
randomisation. This is intentional — it creates a realistic 
scenario for the SRM detection layer (Phase 3) to flag and 
investigate. In production, this tier would have thousands 
of users and hashing would produce near-perfect splits.


**Underpowered experiment (n=2,100):**
With realistic e-commerce conversion rates (1.5–5.5%) and 
only 2,100 users, the experiment generates ~56 purchase events 
— insufficient to reliably detect a 0.5 percentage point lift. 
This is intentional. Phase 3 power analysis will flag the 
experiment as underpowered, and Phase 4's variance reduction 
(CUPED) will demonstrate how sensitivity can be improved 
without increasing sample size. This mirrors a real-world 
scenario where traffic is limited and experimentation 
scientists must work within constraints.
