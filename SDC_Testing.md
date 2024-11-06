## Test Selection for Self-Driving Car Testing: A Tool Competition
---
### Problem Statement
Regression testing for Self-Driving Cars (SDCs) is crucial for ensuring safety and reliability. When changes are made to the SDC software, we need to run tests to verify that existing functionality hasn't been broken. However, running all tests is time-consuming and resource-intensive. The challenge is to select the most relevant test cases that are likely to detect failures while maintaining diversity in the test suite.
### Key Concepts
**Test Selection**: Process of picking only relevant test cases from the test suite for a particular change
**Test Pass/Fail Criteria**:
	**Pass**: Car drives within the lane during the whole simulation
	**Fail**: Car drives off the lane during the simulation
	Tests stop immediately upon failure to save execution costs
### Competition Framework
Uses gRPC for communication between components
Test cases defined as sequences of road points in 2D Cartesian plane
Evaluation metrics:
	Fault Detection: Number of failures detected
	Diversity: Variety in selected test cases
	Time efficiency: Both initialization and selection time
---
## Tools
1. **<u>Sample Selector (Baseline) : </u>**
	Simple random selection strategy
	Serves as a baseline for comparison
	```Python
	Algorithm: Random Test Selection
	Input: Test cases
	Output: Selected test cases

	1. function Select(test_cases):
	    2. selected = []
	    3. for each test in test_cases:
	        4. if random.randint(0, 1) == 1:  # 50% chance
	            5. selected.append(test)
	    6. return selected
	```
1. **<u>ML (Ensemble) Selector</u>**
	Combines Random Forest and Gradient Boosting
	Uses road features and historical data
	```Python
	Algorithm: Ensemble Selection with Random Forest & Gradient Boosting
	Input: Test cases, Historical Data
	Output: Selected test cases

	1. Initialize:
	    2. RandomForestClassifier (50 trees, max_depth=10)
	    3. GradientBoostingClassifier (50 trees, max_depth=5)
	    4. ScikitLearn StandardScaler

	2. function ComputeSelectionScore(test):
	    3. features = extract_road_features(test) # complexity, length, angles, etc
	    4. rf_prob = random_forest.predict_proba(features)
	    5. gb_prob = gradient_boosting.predict_proba(features)
	    6. failure_prob = 0.6 * rf_prob + 0.4 * gb_prob
	    7. diversity_score = compute_diversity(test, selected_tests)
	    8. score = 0.4 * failure_prob + 
	               0.4 * diversity_score +
	               0.2 * complexity_score
	    9. return score

	3. function Select(test_cases):
	    4. selected = []
	    5. for each test in test_cases:
	        6. score = ComputeSelectionScore(test)
	        7. if score > threshold:
	            8. selected.append(test)
	    9. return selected
	```
1. **<u>Transformer Selector</u>**
	Uses sequence modeling with attention
	Processes road geometry as sequential data
	```Python
	Algorithm: Transformer-based Selection
	Input: Test cases, Historical Data
	Output: Selected test cases

	1. Initialize:
	    2. TransformerModel(input_dim=8, hidden_dim=48)
	    3. Position Encoding for sequence data
	    4. Feature extraction pipeline

	2. function ExtractFeatures(road_points):
	    3. points = process_road_points(points)
	    4. features = [
	        lengths / total_length,
	        sin(angles),
	        cos(angles),
	        curvature,
	        cumulative_length,
	        complexity,
	        turn_density,
	        max_angle/pi
	    ]
	    5. return features

	3. function ComputeScore(test):
	    4. features = ExtractFeatures(test)
	    5. failure_prob = transformer_model(features)
	    6. diversity = compute_diversity(test)
	    7. pattern_match = compute_pattern_similarity(test)
	    8. score = 0.7 * failure_prob +
	               0.3 * (0.7 * pattern_match + 0.3 * diversity)
	    9. return score

	4. function Select(test_cases):
	    5. selected = []
	    6. for test in test_cases:
	        7. score = ComputeScore(test)
	        8. if score > min_threshold and len(selected) < max_selections:
	            9. selected.append(test)
	    10. return selected
	```
1. **<u>Graph Neural Network Selector</u>**
	Represents road structure as a graph
	Uses GNN for failure prediction
	```Python
	Algorithm: Graph Neural Network Selection
	Input: Test cases, Historical Data
	Output: Selected test cases

	1. Initialize:
	    2. GNN Model with fault prediction layers
	    3. Feature cache
	    4. Road analysis tools

	2. function ExtractRoadFeatures(points):
	    3. segments = compute_road_segments(points)
	    4. feature_vector = [
	        total_length / 100.0,
	        direct_distance / 100.0,
	        mean_curvature,
	        max_curvature,
	        curvature_std,
	        turn_density,
	        complexity / 5.0,
	        sharp_turns_ratio
	    ]
	    5. return feature_vector

	3. function ComputeScore(test):
	    4. features = ExtractRoadFeatures(test)
	    5. failure_prob = gnn_model(features)
	    6. diversity_score = compute_cosine_diversity(test)
	    7. history_bonus = check_historical_performance(test)
	    8. score = failure_weight * failure_prob * history_bonus +
	               (1 - failure_weight) * diversity_score
	    9. return score

	4. function Select(test_cases):
	    5. min_selections = max(total_tests * 0.25, 40)
	    6. max_selections = min(total_tests * 0.4, 100)
	    7. scores = ComputeScore(test) for each test
	    8. selected = top_k_tests(scores, max_selections)
	    9. return selected
	```
1. **<u>Curvature-based Selector</u>**
	Focuses on geometric properties
	Uses grouping for diversity
	```Python
	Algorithm: Curvature-based Selection with Grouping
	Input: Test cases, Historical Data
	Output: Selected test cases

	1. Initialize:
	    2. Road analyzer for geometry analysis
	    3. Group-based diversity management
	    4. Historical failure patterns

	2. function AnalyzeRoad(points):
	    3. curvature = compute_curvature_profile(points)
	    4. segments = compute_segments(points)
	    5. return RoadAnalysis(
	        curvature_profile,
	        total_length,
	        mean_curvature,
	        max_curvature,
	        turn_count,
	        complexity_score
	    )

	3. function ComputeSelectionScore(test):
	    4. analysis = AnalyzeRoad(test)
	    5. complexity_score = calculate_complexity(analysis)
	    6. failure_bonus = check_historical_failures(test)
	    7. group_penalty = calculate_group_diversity(test)
	    8. recent_penalty = check_recent_selection(test)
	    9. score = complexity_score * 
	               failure_bonus * 
	               group_penalty * 
	               recent_penalty
	    10. return score

	4. function Select(test_cases):
	    5. selected = []
	    6. group_selections = defaultdict(int)
	    7. target_selections = len(test_cases) * selection_ratio
	    
	    # First pass: minimum per group
	    8. for test in sorted_by_score(test_cases):
	        9. if group_selections[test.group] < min_group_selections:
	            10. selected.append(test)
	            11. group_selections[test.group] += 1
	    
	    # Second pass: remaining based on score
	    12. remaining = target_selections - len(selected)
	    13. add_top_scoring_tests(remaining)
	    14. return selected
	```
---
# Results:
## **<u>Summary Table</u>**
|**Selector**|**Selection Count**|**Fault Ratio**|**Diversity**|**Init Time (s)**|**Select Time (s)**|**Time/Fault Ratio**|
|-|-|-|-|-|-|-|
|Sample|90|0.1563|0.0310|0.36|0.18	|252.70|
|ML|96|0.2344|0.0405|1.11	|0.71|187.52|
|Transformer|86|0.1927|0.0366|5.09|0.48	|217.95|
|Graph|76|0.1771|0.0385|1.45|0.30	|209.78|
|Curvature|105|0.2135|0.0370|15.27|4.06	|239.48|
---
## Graphs:
1) **Selection Counts**
![](https://beta.appflowy.cloud/api/file_storage/74d6a614-9ffc-4198-89d5-d3d1b9c8b7f4/v1/blob/12b4cec4%2Db23e%2D4b60%2Dbd7a%2Dbb0bacb154e6/hI8GeuktcD8vUeU46lJphve_U6QtthwbQ7dwU3vbcg8=.png)

2) **Time to Fault Ratio**
	![](https://beta.appflowy.cloud/api/file_storage/74d6a614-9ffc-4198-89d5-d3d1b9c8b7f4/v1/blob/12b4cec4%2Db23e%2D4b60%2Dbd7a%2Dbb0bacb154e6/oWHlLgm6neZxHgZdF2zAYGUnC14ebiEGr2hB8BXdyV0=.png)
3) **Time Performance**
	![](https://beta.appflowy.cloud/api/file_storage/74d6a614-9ffc-4198-89d5-d3d1b9c8b7f4/v1/blob/12b4cec4%2Db23e%2D4b60%2Dbd7a%2Dbb0bacb154e6/xayk4eg7PH2zMgd0_MCFB6uDUcjjHiX24hGgFrVaEdw=.png)

4) **Fault Detection and Diversity**
	![](https://beta.appflowy.cloud/api/file_storage/74d6a614-9ffc-4198-89d5-d3d1b9c8b7f4/v1/blob/12b4cec4%2Db23e%2D4b60%2Dbd7a%2Dbb0bacb154e6/INhYZ3Xnryh93NbCbUx2Tr-nBerP5RsWixEcmpUTmfE=.png)
---
## Key Observations:
1. Trade-off between speed and effectiveness
1. ML-based approaches generally performed better
1. Complex geometric analysis (Curvature) took more time but didn't necessarily improve results
1. Simple random selection (Sample) was fast but ineffective

