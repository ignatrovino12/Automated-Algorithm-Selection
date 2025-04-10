State-of-the-art in Automated Algorithm Selection for Multiple TSP

1. Introduction
The Multiple Traveling Salesman Problem (mTSP) is a generalization of the classic Traveling Salesman Problem (TSP), where multiple salesmen must collectively visit a set of cities while minimizing the total travel cost. This problem has various applications in logistics, transportation, and robotics. Given the complexity and variability of instances, automated algorithm selection can significantly improve solution efficiency by identifying the most suitable algorithm for a given instance. The chosen approach is for the standard mTSP where all the salesman return to the depot and need to visit at least one city.

2. Existing Approaches

2.1 Deterministic Approaches
Branch-and-Bound (B&B): Exhaustively explores the solution space while pruning suboptimal branches.
Dynamic Programming (DP): Solves subproblems recursively but suffers from exponential time complexity.
Branch & Cut: Uses the Branch & Bound search strategy combined with cutting planes to reduce the search space.

2.2 Heuristic Approaches
Greedy Algorithms: Iteratively selects the locally optimal choice (ex. nearest neighbor heuristic).
Minimum Spanning Tree (MST)- Based Methods: These methods construct an MST and convert it into a feasible tour.

2.3 Metaheuristic Approaches
Genetic Algorithms (GA): Uses evolutionary strategies to explore the solution space.
Ant Colony Optimization (ACO): Mimics the foraging behavior of ants to find high-quality solutions.
Particle Swarm Optimization (PSO): Optimizes through the collaboration of particles representing potential solutions.
Simulated Annealing (SA): Mimics the annealing process in metallurgy to escape local optima.
Tabu Search (TS): Uses memory structures to guide the search away from previously explored solutions.

2.4 Hybrid and Machine Learning Approaches
Hybrid Heuristics: Combines multiple heuristics/metaheuristics for better performance.
Hyperheuristics: A higher-level methodology that selects or generates heuristics adaptively.
Anytime Algorithm Selection: Dynamically switches algorithms based on instance characteristics.
AutoML-Based Selection: Uses Machine Learning techniques to predict the best-performing algorithm.

2.5 Chosen Approaches:

2.5 Chosen Approaches

The following approaches were implemented to solve the Multiple Traveling Salesman Problem (mTSP):

1. **Greedy - Shortest Route**
   - **Description**: A simple heuristic that assigns cities to salesmen by iteratively selecting the nearest unvisited city. Each salesman starts and ends at the depot.
   - **Implementation**: 
     - Ensures each salesman visits at least one city.
     - Minimizes the total distance traveled by all salesmen.

2. **KMeans Greedy**
   - **Description**: Combines clustering and greedy heuristics. Cities are grouped into clusters using KMeans, and each cluster is assigned to a salesman. A greedy approach is then used to solve each cluster.
   - **Implementation**:
     - Uses KMeans clustering to divide cities into compact groups.
     - Redistributes cities to avoid empty clusters.
     - Solves each cluster using a greedy algorithm.

3. **Ant Colony Optimization (ACO)**
   - **Description**: A metaheuristic inspired by the foraging behavior of ants. It uses pheromone trails and heuristic information to iteratively improve solutions.
   - **Implementation**:
     - Balances exploration and exploitation to find high-quality solutions.
     - Includes mechanisms to handle stagnation and a time limit of 100 seconds.
     - Updates pheromone trails based on solution quality.

4. **Branch and Cut**
   - **Description**: An exact method that combines Branch and Bound with cutting planes to reduce the search space and solve the problem optimally within a time limit.
   - **Implementation**:
     - Uses linear programming to model the problem.
     - Includes constraints to eliminate subtours and enforce valid routes.
     - Solves the problem using the OR-Tools CBC/GUROBI solver with a 100-second time limit.

Each approach is evaluated based on the following metrics:
- **Total Cost**: The total distance traveled by all salesmen.
- **Time Taken**: The time required to compute the solution.
- **Distance Gap**: The difference between the longest and shortest routes among salesmen.
- **Efficiency**: A metric combining cost and time to evaluate performance.

These approaches provide a mix of heuristic, metaheuristic, and exact methods to address the diverse requirements of mTSP instances.

3. Benchmark Instances
Several benchmark datasets are used to evaluate algorithm performance on mTSP:
TSPLIB: A standard collection of TSP instances.
CVRPLIB: Focuses on capacitated vehicle routing problems.
DIMACS TSP Challenge: Provides large-scale TSP and VRP instances.
Randomly Generated Instances: Used for controlled testing and ML training.

I chose to randomly generate my instances, because the lack of mTSP benchmark datasets due to the focus
being on TSP problems and many such popular benchmarks do bot cover mTSP.

4. Algorithm Selection using Machine Learning

4.1 Feature Selection

Key attributes influencing algorithm performance include:
- Number of cities and salesmen: The size of the problem directly impacts the complexity and the suitability of different algorithms.
- Distance matrix properties: Characteristics such as clustering tendency and edge density can influence the performance of algorithms.
- Graph topology: Features like connectivity and degree distribution provide insights into the structure of the problem instance.
- Solution diversity: The number of feasible solutions within a cost threshold can indicate the difficulty of the problem.

Instance Features Used
The following features are extracted from the problem instances and stored in the `instances` table:
- `average_distance`: The average distance between cities.
- `stddev_distance`: The standard deviation of distances between cities.
- `density`: The density of the graph representing the instance.
- `salesmen_ratio`: The ratio of salesmen to cities.
- `bounding_box_area`: The area of the bounding box enclosing all cities.
- `aspect_ratio`: The aspect ratio of the bounding box.
- `spread`: The spread of cities in the instance.
- `cluster_compactness`: A measure of how compact the clusters of cities are.
- `mst_total_length`: The total length of the Minimum Spanning Tree (MST) for the instance.
- `entropy_distance_matrix`: The entropy of the distance matrix, representing the diversity of distances.

These features are crucial for building predictive models that can recommend the best algorithm for a given instance. By analyzing the relationships between instance features and algorithm performance, we can optimize the selection process and improve problem-solving efficiency.

4.2 Dataset Construction
Collect performance data for multiple algorithms across diverse instances.
Encode problem features to form input vectors.
Label instances with the best-performing algorithm.

4.3 Prediction Models
- Neural Networks: Nhighly suitable for capturing complex, non-linear relationships between instance features and algorithm performance. They can be trained on large datasets to predict the best algorithm for a given instance based on its features.

- LLMs (Large Language Models): Recent advancements in LLMs, such as GPT-based architectures, have introduced new possibilities for algorithm selection. 

By combining Neural Networks and LLMs, we can build robust predictive models that not only recommend the best algorithm but also provide interpretability and adaptability to diverse problem instances.

5. Smol Agents and Autonomous Systems for Algorithm Selection

Recent advancements in lightweight AI agents, such as Smol Agents, have introduced new possibilities for automated algorithm selection in mTSP. These agents can assist in various stages of the selection process, including feature extraction, data preprocessing, and recommendation of algorithms based on historical performance.

Applications of Smol Agents in Algorithm Selection for mTSP:
- Feature Engineering: Smol Agents can assist in identifying and extracting relevant problem characteristics to improve algorithm selection.
- Automated Code Generation: These agents can generate and optimize code for solving mTSP instances, including heuristic and metaheuristic approaches.
- Hyperparameter Optimization: Smol Agents can suggest hyperparameter tuning strategies for Machine Learning models used in algorithm selection.
- Explainability & Interpretability: They can provide insights into why certain algorithms perform better on specific instances.

Autonomous Systems for Algorithm Selection

Autonomous systems leveraging lightweight AI agents could enhance algorithm selection by:
-*Dynamic Algorithm Switching: Adapting algorithm selection in real-time based on problem instance characteristics.
- Multi-Agent Collaboration: Supporting multi-agent decision-making to enable distributed solving of mTSP.
- Integration with Reinforcement Learning: Using feedback mechanisms to continuously improve algorithm recommendations.

While Smol Agents and autonomous systems can support algorithm selection, their current capabilities are limited in directly executing algorithm selection due to the need for empirical evaluation and computational benchmarking.

6. Conclusion
Automated algorithm selection for mTSP is a promising research direction that leverages ML to optimize problem-solving efficiency. Benchmark datasets and systematic feature selection are crucial in building predictive models. Also, LLMs with reinforcement learning and deep learning can be used for dynamic algorithm adaptation.


