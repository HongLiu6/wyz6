import random
ls=[]
k=0
for i in range(50):
    a=random.randint(1,999)
    ls2=[]
    ls2.append(a)
    ls+=ls2
for n in range(50):
    for i in range(n,50):
        for t in range(n,50):
            if ls[i]<=ls[t]:
                ls[i],ls[t]=ls[t],ls[i]
for b in ls:
    k+=1
    print("{:>3}".format(b),end=" ")
    if k==10:
        print()
        k=0
        
    

