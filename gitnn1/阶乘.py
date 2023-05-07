f1="阶乘.txt"
f=open(f1,"w+")
p=1
ls=[]
for t in range(1,101):
        ls.clear()
        g=t*p
        p=t
        ls.append(str(t))
        ls.append("!")
        ls.append("=")
        ls.append(str(g))
        f.write("".join(ls)+"\n")
f.close()
        
        
        
