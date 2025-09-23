# TODO Items

This file tracks pending improvements and technical debt in the Factorama codebase.

## High Priority

### Implement state reversion when residual increases in optimizer
**File:** `src/factorama/sparse_optimizer.cpp:177`  
**Description:** Currently when the residual increases during Gauss-Newton optimization, the optimizer just breaks the loop. It should revert to the previous state to maintain robustness. Note: This primarily affects Gauss-Newton method since Levenberg-Marquardt already has better handling for this case.

## Medium Priority

### Refactor factor placement to use ID-based lookup instead of array indices
**Files:** 
- `src/factorama/factor_graph.cpp:299`
- `src/factorama/factor_graph.cpp:335`

**Description:** Currently using array indices (`factor_placement_[i]`) for factor lookup. Should switch to ID-based lookup for better maintainability and cleaner architecture.

### Move variable increment application from optimizer to factor graph
**File:** `src/factorama/sparse_optimizer.cpp:41`  
**Description:** Variable increment application is currently done in the optimizer. Consider moving this responsibility to the factor graph for better separation of concerns.

### Add logging for inverse range value clipping
**File:** `src/factorama/inverse_range_variable.hpp:50`  
**Description:** Inverse range values are silently clipped to valid bounds. When logger integration is added, include info/warning messages when clipping occurs for better observability.

## Low Priority / Cleanup

### Clean up commented debug code in sparse Jacobian build
**File:** `src/factorama/factor_graph.cpp:436`  
**Description:** Remove or properly integrate commented debug code for Jacobian sanity checks. Currently just adds clutter to the codebase.