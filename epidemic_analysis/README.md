# hw3_p2 explanation
## Functions
**SIR()** :To run a SIR simulation at one time, return the Total_InfectedNodes

**find_best_partner** :Return the best_partner for a fixed `nodeset_left` from candidate set `nodeset_right`

**greedy_find_k()** :To run a greedy algorithm, return the Return f(S) at different `k`

**select_candidate_set()** : To select `n` sets of `size` nodes randomly (can be given a seed)



## A single trial
We can just call **greedy_find_k()**, give `Graph`, `Infection probability`, `node set (Initially infected node)`, and `k`, i.e., the average number of F(s) calculation required. Then it will call **find_best_partner** and find the optimal `u*` then call **SIR()** return the f(S) at different k.


## Entire experiment
Use **select_candidate_set()** to generate some (3) initially infected node sets, then use a for-loop to call **greedy_find_k()** for each set to get the results.