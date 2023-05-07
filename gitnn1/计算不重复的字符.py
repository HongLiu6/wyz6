a=input('请输入一个字符串:')
se=set(a)
t=''
m={}
for i in se:
    if i ==' ':
        continue
    else:
        m[i]=(a.count(i),a.index(i))
n=sorted(m.items(),key=lambda x:(-x[1][0],-x[1][1]))
for z in n:
    t+=z[0]
print('{}{}'.format("不重复的字符是:",t))
