def f1 (pt):
    # f(x) = 1.5(x1) ^ 2 + (x2) ^ 2 - 2(x1)(x2) + 2(x1) ^ 3 + 0.5(x1) ^ 4
    ans = 1.5 * pt[0]**2 + pt[1]**2 - 2 * pt[0] * pt[1] + 2 * pt[0]**3 + 0.5 * pt[0] **4
    return ans
 
def ddx1(pt):                               #Taking Partial derivative with respect to x1

    ansdx1 = 3 * pt[0] - 2 * pt[1] + 6 *pt[0]**2 + 2 * pt[0]**3
    return ansdx1


def ddx2(pt):                               #Taking Partial derivative with respect to x2

    ansdx2 = 2 * pt[1] - 2* pt[0]
    # print("ddx2 = "+str(ansdx2))
    return ansdx2

pt = [[1,-3]]
i = 0 ;
exit = False
max_iterations = 10000
error_threshold = 0.5
Learning_rate = 0.1
fdashx1 = []
fdashx2 = []
# print (fdash)

while (exit == False) :

    fdashx1.append(ddx1(pt[i]))
    fdashx2.append(ddx2(pt[i]))
    newpt = []
    newpt.append(pt[i][0]-Learning_rate * ddx1(pt[i]))
    newpt.append(pt[i][1]-Learning_rate * ddx2(pt[i]))
    i =i+1
    pt.append(newpt)

    if(i>max_iterations):
        exit = True

print('-----------')
print(pt)
list_of_fnval = []

print('Function values : ')
for list in pt:
    fnval = f1(list)
    list_of_fnval.append(fnval)

for a,b,c,d  in zip(pt, list_of_fnval,fdashx1,fdashx2):
    print(a[0],a[1], b,c,d)


print("minimum value found : ")
print(min(list_of_fnval))
