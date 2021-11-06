from p3.curves import *

y_sporty = customer_class(0.35, 11.5, 3, 875*0.5, name='YS')
o_sporty = customer_class(0.35, 11.5, 3, 875*0.5, name='OS')
young = customer_class(0.55, 9, 2, 1000, name='YNS')
old = customer_class(0.7, 6.5, 0.5, 600, name='ONS')

classes = [young, old, y_sporty, o_sporty]
for i in range(len(classes)):
    classes[i].index = i


def obj_fun(user_classes, b, p):
    obj = 0
    for c in user_classes:
        obj += c.clicks(b)*(c.conversion_rate(p)*margin(p)*(c.poisson+1))-c.clicks(b)*cost_per_click(b)
    return obj
