# CHapter-6 Exercises

1. The approximate depth of a decision tree , given its binary is approximate log n , also because its without restrictions , the tree will be less balanced and will round off to around log2(10^6) = 20.
2. A node’s gini impurity is always lower than its parent , because CART is a greedy algorithm , it always tries to minimize gini impurity , and if it can’t it just stops. But in some cases it can have more gini-impurity if the other sibling node has close to 0 impurity because of weighted sum .
3. Yes, decreasing max_depth , would constraint the depth of decision tree , regularizing it/
4. Scaling the features , will not have any effect as decision trees dont care about scaling , and it will simply be a waste of time.
5. Computational complexity is On*(m * log2m)
6. Roughly Doube training time.