import random
import numpy as np
# number of items == number of values==n
# kanapsack capacity == W
# maximum range of values = v
# maximum range of wieghts = w 
# randomly generated itmes array and values array
# return the max weight that can fit in the knapsack


n = int(input('enter the no. of itmes you have : \n'))
W = int(input('enter the maximum capacity of knapsack : \n'))
v = int(input('enter the maximum range of value to store : \n'))
w = int(input('enter the maximum range of the weight to store : \n'))

'''values=[random.randint(0,v) for i in range(0,n)]
wieghts=[random.randint(0,w) for i in range(0,n)]'''
values=np.random.randint(1,v,size=n)
wieghts=np.random.randint(1,w,size=n)

# initialize the memoize matrix 
t = [[-1 for i in range(W+1)] for j in range(n+1)]

def knapsack(wieghts,values,W,n):
    if n==0 or W==0:
        return 0
    if t[n][W] !=-1:
        return t[n][W]
    if wieghts[n-1]<=W:
        t[n][W]=max(values[n-1]+knapsack(wieghts,values,W-wieghts[n-1],n-1),knapsack(wieghts,values,W,n-1))
        return t[n][W]
    elif wieghts[n-1]>W:
        t[n][W]=knapsack(wieghts,values,W,n-1)
        return t[n][W]


print(values)
print(wieghts)
print(knapsack(wieghts,values,W,n))



