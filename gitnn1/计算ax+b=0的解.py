a=eval(input("请输入a的值"))
b=eval(input("请输入b的值"))
if a == 0:
    if b==0:
        print("该方程有无数个解")
    else:
        print("该方程无解")
else:
    x=-(b/a)
    print("该方程的解是",x)
    
