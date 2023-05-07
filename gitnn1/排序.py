a=ord(input("请输入开始字母是:"))
b=ord(input("请输入开始字母是:"))
if a>b:
    a,b=b,a
for i in range(b-a+1):
    c=""
    for d in range(b-a+1):
        e=a+d+i
        if e>b:
            f=e-b-1
            e=a+f
        c=c+chr(e)
    print(c)
