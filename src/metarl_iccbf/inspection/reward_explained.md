Looking at the MORL block in inspectionEnvNoisy.py, there are actually two things:

1. The cos_angle shaping term (line 886):


coverage_reward += 0.01 * max(cos_angle, 0.0)
A per-step potential that rewards angular alignment toward the largest uninspected cluster — added to encourage orbital maneuvering rather than hovering. Incentivizes inspection progress.

2. The -60.0 terminal penalty on coverage (line 895):


if terminated and safety_penalty > 0:
    r_vec[0] -= 60.0
The comment explains it: at high fuel weights, the agent could "commit suicide" (deliberately terminate early to avoid fuel cost). The -60 on coverage makes that never worth it regardless of the weight vector.

Does either affect the proposed changes? No — both are already baked into r_vec[0] before any weight is applied. The normalization of min_h and the change to which objective is fixed are orthogonal to both of these. They'll keep working exactly as intended.