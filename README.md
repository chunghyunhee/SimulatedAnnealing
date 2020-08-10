# Simulated Annealing for HPO

1. Flowchart
![image](https://user-images.githubusercontent.com/49298791/89750327-79ebba80-db06-11ea-9297-2660e4008152.png)

2. Algorithm 
![image](https://user-images.githubusercontent.com/49298791/89750310-64769080-db06-11ea-972f-b9f08ad3e6b2.png)

3. method to implement 
	1. Randomly choose a value for all hyperparameters and treat it as current state and evaluate model performance
	2. Alter the current state by randomly updating value of one hyperparameters by selecting a value in the immediate neighborhood(randomly) to get neighboring state.
	3. If the combination is randomly visited, repeat step 2 until a new combination is generated
	4. Evaluate model performance on the neighboring state
	5. Compare the model perforance of neightborsing state to the current state and decide whether to accept the nerighboring state as current state or reject it based on some criteria (explained right side)
Based on the result 5, repeat steps 2 though 5.

4. acceptance criteria
- If the performance of neighboring state is better than current state : Accept
- If the performance of neighboring state is worse than current state : Accept with the exponential prob.

5. reference sites
- algorithm paper : https://arxiv.org/pdf/1906.01504.pdf
- overall procedure : https://learnwithpanda.com/2020/04/04/python-code-of-simulated-annealing-optimization-algorithm/?opanda_confirm=1&opanda_lead=86&opanda_code=ba06a3e354634378be47ebb648ab7369
