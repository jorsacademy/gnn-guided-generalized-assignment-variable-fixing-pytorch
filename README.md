# GNN-Guided Generalized Assignment Variable Fixing

A from-scratch PyTorch graph-neural-network project for using learned predictions as **primal guidance** inside an exact Generalized Assignment Problem (GAP) solver.

The neural network does not replace the optimizer.

```text
GAP instance
   ↓
bipartite machine-job graph
   ↓
message-passing GNN
   ↓
job → machine probabilities
   ↓
validation-calibrated high-confidence fixing
   ↓
exact HiGHS MILP repair
```

The central research question is whether a GNN can identify sufficiently reliable assignment variables to reduce the residual MIP without surrendering exact feasibility repair.

## GAP formulation

For machines `i` and jobs `j`, decision variables are `x[i,j] ∈ {0,1}`. The model minimizes total assignment cost subject to exactly one machine per job and machine capacity limits. The downstream problem is solved with `scipy.optimize.milp` / HiGHS.

## Synthetic benchmark

Instances contain heterogeneous machine efficiency, job-specific resource requirements, machine cost levels, nonlinear machine/job interactions, preference noise and tight but feasible capacities. A hidden balanced assignment is used only to construct guaranteed-feasible capacities; it is not used as a learning label. Labels come from independently solving each generated GAP to MILP optimality.

## Bipartite graph

The model uses machine nodes, job nodes and assignment edges. Machine features summarize capacity pressure and assignment costs; job features summarize relative cost/resource difficulty; edge features contain normalized assignment cost and resource/capacity information. No optimal-solution label is included in the graph features.

## Pure-PyTorch message passing

The implementation does not depend on PyTorch Geometric. Each layer forms edge messages from machine, job and edge embeddings, aggregates messages back to both partitions, applies residual updates and LayerNorm, and finally scores every machine-job edge. A softmax across machines gives `P(machine i | job j, instance)`.

Training uses exact MILP assignments and per-job cross entropy.

## Validation-calibrated fixing

A raw softmax threshold is not assumed calibrated. The repository chooses a confidence threshold on a disjoint validation set. Default development settings target 95% fixing precision at at least 5% coverage. If this cannot be achieved, the conservative fallback fixes no jobs.

For every accepted prediction, the corresponding assignment is fixed and uncertain jobs remain free.

## Exact repair with feasibility recovery

High-confidence fixings can still collectively overload a machine. The residual MILP is solved exactly. If the fixing set is infeasible, the least-confident fixing is relaxed and the MILP is re-solved until feasibility is restored.

The repair MILP is exact **conditional on the surviving fixings**. It is not necessarily globally optimal for the original GAP because a wrong but feasible fixing may exclude the original optimum.

## Oracle fixing control

For each test instance, an oracle control fixes the same set of jobs selected by GNN confidence but assigns those jobs according to the known exact optimum. This separates the benefit of reducing residual problem size from the damage caused by incorrect learned fixings.

## Greedy baseline

A multi-start capacity-aware greedy heuristic is included as a non-exact baseline. It is not described as a MILP warm start because SciPy `milp` does not expose a general primal-start interface.

## Independent exact oracle

On tiny regression fixtures, every possible machine assignment is exhaustively enumerated and capacity-infeasible combinations are discarded. The exhaustive optimum must match HiGHS to numerical tolerance.

## Development benchmark

Seed-42 development configuration:

```text
machines                  6
jobs                     18
training instances       64
validation instances     16
test instances           16
GNN epochs               20
hidden dimension         48
message-passing layers    3

target fixing precision 95%
minimum fixing coverage   5%
margin threshold          0.05
```

Observed result:

```text
best validation assignment accuracy   78.8%
calibrated confidence threshold        0.807
validation fixing precision            95.2%
validation fixing coverage             43.4%

test assignment accuracy               78.5%
mean job confidence                     0.743

initially fixed jobs                    8.25 / 18
finally fixed jobs                      8.06 / 18
wrong initial fixes                     0.250 / instance
relaxed infeasible fixings              0.188 / instance

GNN repair objective gap                0.045%
greedy heuristic objective gap          0.669%
```

The objective gap is relative to the exact vanilla MILP optimum.

Solver behavior in the same local development run:

```text
mean HiGHS MIP nodes
vanilla MILP       0.81
GNN repair         0.88
oracle fixing      0.81

mean solve time
vanilla MILP       0.0064 s
GNN repair         0.0071 s
oracle fixing      0.0049 s
```

This benchmark does **not** demonstrate a learned solver speedup. These small synthetic GAPs are usually solved at or near the root node, so learned fixing overhead can exceed any reduction benefit. This negative result is retained intentionally: classification accuracy does not imply branch-and-bound acceleration.

## Metrics

The experiment reports assignment accuracy, confidence, validation fixing precision/coverage, fixed-job counts, wrong/relaxed fixes, exact-repair objective gap, greedy gap, HiGHS node counts and vanilla/repair/oracle-fixing wall time.

## Tests

The regression suite checks:

- HiGHS MILP vs full enumeration on a tiny GAP;
- post-solve feasibility;
- deterministic instance generation;
- bipartite GNN tensor shapes and gradient flow;
- exact MILP training labels;
- validation confidence calibration;
- infeasible learned-fixing relaxation;
- end-to-end GNN training + exact repair.

## Run

```bash
pip install -r requirements.txt
python gnn_gap_variable_fixing.py --self-test
python -m unittest discover -s tests -v
```

Development experiment:

```bash
python gnn_gap_variable_fixing.py \
  --seed 42 \
  --machines 6 \
  --jobs 18 \
  --train-instances 64 \
  --validation-instances 16 \
  --test-instances 16 \
  --epochs 20 \
  --batch-size 16 \
  --hidden-dim 48 \
  --layers 3 \
  --target-fix-precision 0.95 \
  --minimum-fix-coverage 0.05
```

## Exactness and claims

Exact statements:

- training labels are produced by HiGHS solutions reported optimal;
- vanilla test GAPs are solved to HiGHS optimality;
- tiny regression instances are independently checked by exhaustive enumeration;
- repair is solved exactly for the residual MILP conditional on surviving fixings.

Not claimed:

- GNN predictions are exact;
- learned fixing preserves the original global optimum;
- the current benchmark accelerates HiGHS;
- classification accuracy implies branch-and-bound improvement;
- synthetic timings transfer to production MIPs.

The point of the project is to expose the full learned-primal-guidance pipeline and measure both its benefits and its failure modes.
