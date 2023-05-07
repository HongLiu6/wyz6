import random
t=10000
for i in range(100000):
    a=int(6*random.random()+1)
    b=int(6*random.random()+1)
    c=int(6*random.random()+1)
    d=int(6*random.random()+1)
    if a!=b and a!=d and a!=c:
        t=t-1
    elif a==b and a!=c and a!=d:
        t=t+1
    elif a==c and a!=b and a!=d:
        t=t+1
    elif a==d and a!=b and a!=c:
        t=t+1
    elif a==b and a==c and a!=d:
        t=t+2
    elif a==b and a==d and a!=c:
        t=t+2
    elif a==c and a==d and a!=b:
        t=t+2
    else:
        t=t+3
print(t)
