# Local planner path library

These are robot-relative motion primitives, independent of the live environment.
The planner selects and transforms them using current obstacles and goals.

| Asset | Rows | Meaning |
| --- | ---: | --- |
| `startPaths.ply` | 707 | 101 points per initial-direction group, 7 groups |
| `paths.ply` | 103243 | 301 points per candidate, 343 candidates |
| `pathList.ply` | 343 | Endpoint, path ID and group ID |
| `correspondences.txt` | 72611 | Grid-to-path collision associations |

The original model is in `path_generator.m`: radial sampling 0.01 m to 3 m,
initial angles -27 to 27 degrees in 9-degree steps, branch scale 0.65.
The grid is 161 x 451 with spacing 0.02 m, offsets (3.2, 4.5) m and
collision search radius 0.45 m. Keep these aligned with `localPlanner.cpp`.

From the repository root, with NumPy and SciPy installed:

```sh
python3 scripts/generate_local_planner_paths.py
```

This headless reproduction uses the same not-a-knot cubic spline and path ID
ordering as the MATLAB script. Before writing the PLY files it checks every
grid/path association against the existing correspondence table. On 2026-09-23,
all 72,611 x 343 associations matched on Orin with SciPy 1.8.0. The existing
table was retained. Geometry changes require regenerating and validating the
whole library together.

All four runtime files are versioned and installed by the package's existing
`install(DIRECTORY paths ...)` rule. CMake checks their presence at configure
time. Generation dependencies are needed only when regenerating assets.
