
import numpy as np

def f_String(x):
    temp = 0
    sig  = 1e5
    constraint_ueq = [
        lambda x: 1 - x[1]**3*x[2]/(71785*x[0]**4),
        lambda x: (4*x[1]**2-x[0]*x[1])/(12566*(x[1]*x[0]**3-x[0]**4)) + 1/(5108*x[0]**2)-1,
        lambda x: 1-140.45*x[0]/(x[1]**2*x[2]),
        lambda x: (x[0]+x[1])/1.5-1
        ]
    for constraint in constraint_ueq:
        if constraint(x)>0:
            temp+=constraint(x)**2*sig
    return (x[2]+2)*x[1]*x[0]**2+temp

n_dim_String = 3


lb_String = np.array([0.05, 0.25, 2.00])
ub_String = np.array([2.00, 1.30, 15.0])




def f_Himmelblau(x):
    def g1(x):
        return 85.334407 + 0.0056858*x[1]*x[4] + 0.0006262*x[0]*x[3] - 0.0022053*x[2]*x[4]

    def g2(x):
        return 80.51249 + 0.0071317*x[1]*x[4] + 0.0029955*x[0]*x[1] + 0.0021813*x[2]**2

    def g3(x):
        return 9.300961 + 0.0047026*x[2]*x[4] + 0.0012547*x[0]*x[2] + 0.0019085*x[2]*x[3]

    constraint_ueq = [
        lambda x: g1(x) - 92,
        lambda x: -g1(x),
        lambda x: g2(x) - 110,
        lambda x: 90 - g2(x),
        lambda x: g3(x) - 25,
        lambda x: 20 - g3(x)
        ]
    
    temp = 0
    sig  = 1e100
    for constraint in constraint_ueq:
        if constraint(x)>0:
            temp+=constraint(x)**2*sig
    return 5.3578547*x[2]**2 + 0.8356891*x[0]*x[4]+37.293239*x[0]-40792.14+temp




n_dim_Himmelblau = 5

lb_Himmelblau = np.array([78 , 33, 27, 27, 27])
ub_Himmelblau = np.array([102, 45, 45, 45, 45])