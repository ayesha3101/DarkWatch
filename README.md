# DarkWatch

An AI patrol planner for hidden fishing boats. Built for the AL-2002 Artificial Intelligence Lab project (Fall 2026).

> Work in progress. This README describes what we are building and will be updated as the code is added.

## What it is

Some illegal fishing boats switch off their AIS tracking so nobody can see them. A coast guard only has a few patrol boats with limited fuel and a very large sea to cover, so the hard question is where to send the boat.

DarkWatch keeps a probability map (a belief) of where each hidden boat might be, plans fuel-limited routes with A* search, and uses Monte Carlo tree search to decide where to go next. A local web page shows the map as the belief spreads out over time and collapses when a boat is found.

```
the belief spreads from the last known position, then collapses when the boat is found

 . . . . . . . . . . . .          . . . . . . . . . . . .
 . . . . ░ ░ . . . . . .          . . . . . . . . . . . .
 . . . ░ ▒ ▒ ░ . . . . .   --->   . . . . . . . . . . . .
 . . . ░ ▒ █ ▒ ░ . . . .          . . . . . . ◆ . . . . .     ◆ = boat found
 . . . . ░ ▒ ░ . . . . .          . . . . . . . . . . . .
```

## How it works

1. The patrol spots boats near it (not always) and receives AIS from boats that are still transmitting.
2. A Bayesian filter updates the probability map. Searching a cell and finding nothing lowers the chance of a boat being there.
3. MCTS tries many imagined futures through the belief and picks a promising area.
4. A* finds a cheap route to that area, using a heuristic we write ourselves.
5. This repeats until the fuel runs out.

BFS, UCS, A*, MCTS and the belief filter are our own code. We compare each patrol strategy against simple baselines (random, fixed sweep, go to the last seen position) with a fixed seed so results can be repeated.

## Status

- [x] Problem formulated and proposal written (see `docs/`)
- [ ] Data loading and analysis
- [ ] BFS, UCS and A* search
- [ ] Bayesian belief filter
- [ ] Patrol agents and baselines
- [ ] Monte Carlo tree search
- [ ] Naive Bayes vs decision tree classifier
- [ ] Experiment runner, results table and charts
- [ ] Local web interface

## Planned features

| Feature | Lab |
|---|---|
| Object-oriented design, seeded randomness | 01-02 |
| PEAS description and agent type | 03 |
| BFS and UCS with node counts | 04 |
| A* with our own heuristic | 05 |
| Monte Carlo tree search | 08 |
| Bayesian belief filter | 09 |
| Data cleaning and summaries with Pandas | 10 |
| Naive Bayes vs decision tree classifier | 11 |
| Experiment runner with results table and charts | - |
| Local web interface that works offline | - |

## Running it

These are the planned commands and will work once the code in `src/` is added. Needs Python 3.10 or newer.

```bash
python -m venv .venv
# Windows:   .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

Web interface:

```bash
python src/server.py
# then open http://127.0.0.1:5000
```

Console version:

```bash
python src/main.py
```

Reproduce the experiments and charts:

```bash
python src/experiments.py
```

It runs offline. If the real data files are missing, it says so and uses a built-in simulator.

## Results

Results will be added after the experiments are run. Numbers and charts will be in `outputs/` and in the report in `docs/`.

## Project layout

```
DarkWatch/
├── README.md
├── requirements.txt
├── docs/              proposal and report (PDF and Markdown)
├── src/               source code
│   ├── data.py        loading, cleaning and caching data
│   ├── belief.py      Bayesian belief filter
│   ├── search.py      BFS, UCS and A*
│   ├── mcts.py        Monte Carlo tree search
│   ├── agents.py      patrol agents and baselines
│   ├── classifier.py  Naive Bayes vs decision tree
│   ├── experiments.py experiment runner
│   ├── server.py      local Flask server for the web page
│   └── web/           HTML, CSS and JavaScript
├── data/              raw/ and processed/
└── outputs/           figures/ and results/
```

This is the planned layout and may change as we build.

## Data

- Global Fishing Watch (https://globalfishingwatch.org) for fishing activity. Used for non-commercial, academic purposes and credited to them.
- Danish Maritime Authority AIS data (https://www.dma.dk) for vessel tracks, used for the movement model, the classifier and belief tests.
- A built-in simulator as a backup when real data is missing.

Raw data is not stored in this repository. API keys are read from environment variables and are never committed.

## Limitations

- Apparent fishing is not illegal fishing. Most fishing is legal, so the tool only suggests where to look and never accuses anyone.
- It is a prototype, not an operational tool. It ignores weather, jurisdiction, crew limits and real intelligence, and no agency has tested it.
- The Danish tracks come from other waters, so they are only used to learn how boats move and to test the belief filter.
- Boats that hide on purpose are simulated, because real offender behaviour is not in any dataset.
- A system like this could unfairly target small-scale fishers, so it should support human decisions, not replace them.

## Credits

Data from Global Fishing Watch and the Danish Maritime Authority. Libraries: NumPy, Pandas, scikit-learn, Matplotlib, Flask.
