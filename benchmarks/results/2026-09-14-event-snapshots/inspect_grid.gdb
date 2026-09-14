# SPDX-License-Identifier: MIT
set startup-with-shell off
set pagination off
set env OMP_NUM_THREADS 1
set env OMP_PROC_BIND false
printf "GRID_OBJECT bytes=%zu\n", sizeof(mango::Grid<double>)
break mango::AmericanOptionSolver::solve
commands
silent
printf "GRID steps=%zu space=%zu\n", this->grid_config_.second.n_steps_, this->grid_config_.first.n_points_
continue
end
run --benchmark_filter=^BM_EventManualBuild/[13] --benchmark_min_time=1x --benchmark_min_warmup_time=0
