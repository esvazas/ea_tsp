# How to solve TSP?
Implementation of Evolutionary Algorithm to solve famous Travelling Salesman Problem.
The code:
* was written for Genetic Algorithms and Evolutionary Computing course at KU Leuven, 2020. 
* was considered among Top-15 percent best implementations for the competition among ~100 students.
* was implemented to solve large TSP tasks (~1000 cities) within 5 minutes timeframe.
The full project report containing the details of implementation, comparison of results with respect to greedy heuristic solutions, and possible modifications can be accessed in report.pdf file. Implementation was initially constrained only to a single python file to allow for comparison on unseen datasets with respect to other students.
 
## Experiments content
* data - directory containing two example datasets which include 29 and 100 cities.
* solver - directory containing EA implementation and Reporter class.  
* result - directory containing solutions.

## Requirements
```
numpy-1.19.2
python-3.7.10
```

## Usage
After you clone this repo, go to its root directory and start the algorithm for one of the example datasets:
```bash
python main.py
```
