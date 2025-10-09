import random
from math import sqrt

c1, c2 = 1, 1


def fitness(x):
    return -x**2 + 5*x + 20


def init():
    n = int(input("Enter no. of particles: "))
    v = [0 for i in range(n)]
    x = list(map(float, input("Enter positions of particles:").split()))
    p = x.copy()
    fp = [fitness(xi) for xi in x]
    return n, v, fp, p, x


def find(n, fp, p):
    max_fitness = float('-inf')
    pos = -1
    for i in range(n):
        if fp[i] > max_fitness:
            max_fitness = fp[i]
            pos = i
    return pos


def update(n, v, fp, p, x, max_pos):
    r1, r2 = sqrt(random.random()), sqrt(random.random())

    for i in range(n):
        v[i] = v[i] + c1 * r1 * (p[i] - x[i]) + c2 * r2 * (p[max_pos] - x[i])
        x[i] = x[i] + v[i]

    for i in range(n):
        fp[i] = fitness(x[i])
        if fp[i] > fitness(p[i]):
            p[i] = x[i]


def print_state(v, fp, p, x):
    print(f'''
    {x}
    {p}
    {v}
    {fp}
    ''')


n, v, fp, p, x = init()
print_state(v, fp, p, x)
max_pos = find(n, fp, p)
gbest = p[max_pos]

while True:
    update(n, v, fp, p, x, max_pos)
    max_pos = find(n, fp, p)
    if fitness(gbest) == fitness(p[max_pos]):
        break
    print_state(v, fp, p, x)
    gbest = p[max_pos]

print(f"Global Best Solution: {gbest} with fitness: {fitness(gbest)}")
