a=eval(input("请输入行数:"))
b=" "
c="* "
for i in range(a):
    if i==0:
        t=2*a-2
        print(t*b+a*c)
    elif i==(a-1):
        print(a*c)
    else:
        n=2*(a-i)-2
        m=2*(a-1)-2
        print(n*b+c+m*b+c)
