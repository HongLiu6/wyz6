fr=open("2015年5月高三模拟考成绩.csv","rt")
import time
a={}
b=0
for h in fr.readlines():
    h=h.replace("\n","")
    c=h.split(",")
    b+=1
    if j[0][0]=="3":
        d=h.split(",")
        b=eval(d[3])+eval(d[4])+eval(d[5])+eval(d[6])+eval(d[7])+eval(d[8])
        a[d[0]]=(d[1],d[2],b)
fr.close()
c=[]
e=sorted(a.items(),key=lambda x: x[1][2],reverse=True)
for i in range(5):
    c.append(e[i])

ls="{:<8}\t{:<8}\t{:<8}\t{:<8}"
print("总分前五名的学生情况")
print(ls.format("名次","班别","姓名","总分"))
print("-"*50)
for i in c:
    j=1
    print(ls.format(j,i[1][0],i[1][1],i[1][2],))
    j+=1
