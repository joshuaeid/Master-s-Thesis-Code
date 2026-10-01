# LineCover

Code for the master's thesis *Modeling and Heuristic Solution of an Optimal Line Cover Problem*: modelling bus systems as sets of positional circuits, evaluating the ensemble average of the service time, and approximating optimal systems with heuristics. The code was tested with Python 3.13 and the packages listed in `requirements.txt` are required. Gurobi is only needed for the phase harmonisation.

## Layout

- `LineCover/SystemKit` - graph, buses, bus systems, the ensemble average and its Monte Carlo estimate, and the plotting helpers used for the figures of the thesis.
- `LineCover/InitialHeuristic` - construction of a feasible start solution.
- `LineCover/RouteHeuristic` - the three route moves, the local search and the iterated local search.
- `LineCover/PhaseHarmonisationHeuristic` - the coincidence arrangement and the region hopping.
- `LineCover/Metaheuristics` - `olc_four_stages` for small instances and `olc_route_mc` for instances on which the exact ensemble average is out of reach.
- `notebooks` - the test runs of the thesis.

## Test Notebooks

Jupyter has to be started in the `notebooks` folder, since the notebooks add the repository root to the path relative to their own location.

`OLCFourStages_Test_Topologies.ipynb` and `OLCFourStages_Test_3x3_Grid.ipynb` contain the test runs of the four-stage heuristic on small instances, `OLCRouteMC_Test_22x22_Grid.ipynb` and `OLCRouteMC_Test_32x32_Grid.ipynb` every computation behind Sections 6.6.3 and 6.6.4. Each notebook keeps its results in a folder of the same name.