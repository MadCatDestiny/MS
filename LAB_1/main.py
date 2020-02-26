import numpy as np
from matplotlib import pyplot as plt
import math

def clear_print(arr):
    max_len = max([len(str(e)) for r in arr for e in r])
    for row in arr:
        print(*list(map('{{:>{length}}}'.format(length=max_len).format, row)))

N = 200
v = 83  # int(input('Variant = '))
n = 5 + v % 16
p = 0.3 + v * 0.005
lamb = 0.5+0.01 * v
print('p = {}\nn = {}\nlamb = {}'.format(p, n, lamb))

binom_not_sort = [1, 7, 6, 6, 5, 5, 5, 5, 6, 6, 5, 5, 5, 8, 8, 7, 6, 7, 5, 5, 5, 6, 4, 7, 7, 7, 5, 5, 4, 7, 5, 5, 5, 5, 6, 5, 5, 3, 7, 6, 5, 5, 8, 6, 6, 5, 5, 2, 6, 6, 6, 7, 6, 5, 6, 6, 7, 6, 5, 4, 4, 6, 6, 5, 8, 7, 6, 5, 6, 6, 6, 5, 6, 7, 5, 7, 6, 4, 3, 7, 8, 7, 6, 6, 6, 6, 5, 5, 4, 7, 4, 5, 6, 5, 6, 4, 7, 5, 4, 6, 5, 7, 3, 5, 5, 6, 4, 7, 5, 6, 6, 7, 6, 7, 5, 5, 5, 6, 4, 7, 6, 5, 7, 5, 8, 5, 6, 6, 5, 6, 4, 5, 6, 7, 5, 4, 6, 7, 6, 6, 7, 5, 4, 7, 6, 7, 4, 7, 4, 6, 6, 8, 6, 6, 8, 7, 4, 7, 6, 6, 6, 8, 2, 5, 6, 5, 7, 6, 7, 7, 7, 5, 4, 4, 5, 5, 6, 6, 8, 6, 7, 4, 6, 7, 8, 4, 4, 5, 7, 4, 7, 6, 6, 6, 5, 8, 6, 5, 6, 5]
# np.random.binomial(n, p, 200)

geom_not_sort = [0, 1, 0, 1, 1, 0, 0, 3, 4, 2, 2, 0, 2, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 2, 0, 1, 1, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 1, 0, 0, 1, 0, 5, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0, 1, 0, 0, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 2, 0, 0, 1, 0, 1, 1, 0, 0]
# np.random.geometric(p, 200)

paus_not_sort = [1, 1, 0, 1, 2, 1, 1, 0, 1, 2, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 3, 1, 1, 1, 0, 1, 1, 3, 1, 0, 1, 1, 0, 0, 3, 2, 1, 2, 2, 2, 0, 0, 4, 0, 0, 0, 2, 2, 0, 1, 1, 0, 4, 2, 3, 2, 1, 0, 1, 1, 2, 2, 0, 0, 2, 0, 1, 1, 3, 3, 3, 2, 2, 0, 3, 0, 2, 2, 0, 2, 2, 0, 1, 2, 0, 1, 1, 1, 2, 3, 0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 4, 0, 0, 2, 0, 1, 3, 0, 1, 1, 0, 0, 1, 2, 0, 1, 0, 0, 1, 1, 0, 2, 1, 0, 1, 0, 2, 1, 3, 2, 1, 3, 1, 2, 0, 3, 3, 3, 1, 1, 2, 4, 0, 0, 2, 2, 1, 3, 1, 1, 2, 2, 0, 0, 2, 2, 2, 3, 1, 1, 1, 0, 2, 1, 1, 1, 3, 1, 3, 2, 1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 3, 4, 2, 1, 0, 1, 1, 1, 0, 2, 2, 2, 0, 3, 1, 0, 1]
# np.random.poisson(lamb,200)

binom = np.sort(binom_not_sort)
geom = np.sort(geom_not_sort)
puas = np.sort(paus_not_sort)

######
# BIN
######
binom_seq = 0
bin_stat_seq = np.zeros((3, n), dtype=float)

for i in range(1, n+1):
    binom_seq = 0
    w = 0
    for k in range(N):
        if binom[k] == i:
            binom_seq += 1
        else:
            continue
    w = binom_seq / N
    bin_stat_seq[0, i-1] = int(i)
    bin_stat_seq[1, i-1] = int(binom_seq)
    bin_stat_seq[2, i-1] = w


print('\nbinomial static sequence: ')
clear_print(bin_stat_seq.tolist())
binom_x = bin_stat_seq[0]
binom_n = bin_stat_seq[1]
binom_w = bin_stat_seq[2]

fig, axs = plt.subplots(3, 2)
axs[0, 0].plot(binom_x, binom_w)
axs[0, 0].set(xlabel='w', ylabel='x')
axs[0, 0].set_title('binomial distribution')
axs[0, 0].grid(linestyle='--', linewidth=1)
axs[0, 0].set_xbound(1,8)
axs[0, 0].set_ybound(0,binom_w[-1])
axs[0, 0].set_xticks(range(1,9))
binom_w.sort()
axs[0, 0].set_yticks([(i * int(i % 2 == 0))/100 for i in range(0,int(binom_w[-1]*100)+5,6)])

axs[0, 1].hlines(np.cumsum(binom_w),range(int(binom_x[-1])),range(1,int(binom_x[-1])+1))
axs[0, 1].set_xbound(1,8)
axs[0, 1].set_ybound(0,1.1)
axs[0, 1].set_xticks(range(1,int(binom_x[-1]+1)))
axs[0, 1].set_yticks([(x*(x%2==0))/10 for x in range(11)])
axs[0, 1].grid(linestyle='--', linewidth=1)

##
q = 1 - p
print('\nBinomial characteristics:')
print('Мат ожидание: {:.5f}'.format(float(n*p)))
print('Дисперсия: {:.5f}'.format(float(n*p*q)))
print('Среднее квадратическое отклонение: {:.5f}'.format(float(math.sqrt(n*p*q))))
print('Мода: {:.5f}'.format(float((n+1)*p if int((n+1)*p) != float((n+1)*p) else ((n+1)*p) - 0.5)))
print('Медиана: {:.5f}'.format(round(n*p)))
print('Коэф ассиметрии: {:.5f}'.format(float((q-p)/math.sqrt(n*p*q))))
print('Коэф эксцесса : {:.5f}'.format(float((1 - 6*p*q)/(n*p*q))))
##

######
# GEOM
######
geom_seq = 0
geom_stat_seq = np.zeros((3, max(geom)+1), dtype=float)
for i in range(min(geom), max(geom)+1):
    geom_seq = 0
    w = 0
    for k in range(N):
        if geom[k] == i:
            geom_seq += 1
        else:
            continue
    w = geom_seq / N
    geom_stat_seq[0, i] = int(i)
    geom_stat_seq[1, i] = int(geom_seq)
    geom_stat_seq[2, i] = w

print('\ngeometric static sequence: ')
clear_print(geom_stat_seq.tolist())
geom_x = geom_stat_seq[0]
geom_n = geom_stat_seq[1]
geom_w = geom_stat_seq[2]

axs[1, 0].plot(geom_x, geom_w)
axs[1, 0].set(xlabel='w', ylabel='x')
axs[1, 0].set_title('geometric distribution')
axs[1, 0].grid(linestyle='--', linewidth=1)
axs[1, 0].set_xbound(0, max(geom_x))
axs[1, 0].set_ybound(0, geom_w[-1])
axs[1, 0].set_xticks(range(0, int(max(geom_x)+1)))
s_geom_w = geom_w.copy()
s_geom_w.sort()
axs[1, 0].set_yticks([(i * int(i % 2 == 0))/100 for i in range(0,int(s_geom_w[-1]*100)+11, 20)])

axs[1, 1].hlines(np.cumsum(geom_w), range(int(geom_x[-1])+1), range(1,int(geom_x[-1])+2))
axs[1, 1].set_xbound(0, max(geom_x))
axs[1, 1].set_ybound(0, 1.1)
axs[1, 1].set_xticks(range(1, int(geom_x[-1]+1)))
axs[1, 1].set_yticks([(x*(x % 2 == 0))/10 for x in range(11)])
axs[1, 1].grid(linestyle='--', linewidth=1)

##
print('\nGeometric characteristics:')
print('Мат ожидание: {:.5f}'.format(float(q/p)))
print('Дисперсия: {:.5f}'.format(float(q/p**2)))
print('Среднее квадратическое отклонение: {:.5f}'.format(float(math.sqrt(q)/p)))
print('Мода: 0')
print('Медиана: {:.5f}'.format(-(math.log1p(2)/math.log1p(q))
                               if int(math.log1p(2)/math.log1p(q)) == float(math.log1p(2)/math.log1p(q))
                               else -(math.log1p(2)/math.log1p(q)) - 0.5))
print('Коэф ассиметрии: {:.5f}'.format(float((2-p)/math.sqrt(q))))
print('Коэф эксцесса : {:.5f}'.format(float((6 + ((p**2)/q)))))
##

######
# PUAS
######
puas_seq = 0
puas_stat_seq = np.zeros((3, max(puas)+1), dtype=float)

for i in range(min(puas), max(puas+1)):
    puas_seq = 0
    w = 0
    for k in range(N):
        if puas[k] == i:
            puas_seq += 1
        else:
            continue
    #if binom_seq > 0:
    w = puas_seq / N
    puas_stat_seq[0, i] = int(i)
    puas_stat_seq[1, i] = int(puas_seq)
    puas_stat_seq[2, i] = w

print('\npuanson static sequence: ')
clear_print(puas_stat_seq.tolist())
puas_x = puas_stat_seq[0]
puas_n = puas_stat_seq[1]
puas_w = puas_stat_seq[2]

axs[2, 0].plot(puas_x, puas_w)
axs[2, 0].set(xlabel='w', ylabel='x')
axs[2, 0].set_title('puasetric distribution')
axs[2, 0].grid(linestyle='--', linewidth=1)
axs[2, 0].set_xbound(0, max(puas))
axs[2, 0].set_ybound(0, puas_w[-1])
axs[2, 0].set_xticks(range(0, max(puas)+1))
s_puas_w = puas_w.copy()
s_puas_w.sort()
axs[2, 0].set_yticks([(i * int(i % 2 == 0))/100 for i in range(0, int(s_puas_w[-1]*100)+7, 6)])

axs[2, 1].hlines(np.cumsum(puas_w), range(int(puas_x[-1])+1), range(1,int(puas_x[-1])+2))
axs[2, 1].set_xbound(0, max(puas_x))
axs[2, 1].set_ybound(0, 1.1)
axs[2, 1].set_xticks(range(1, int(puas_x[-1]+1)))
axs[2, 1].set_yticks([(x*(x % 2 == 0))/10 for x in range(11)])
axs[2, 1].grid(linestyle='--', linewidth=1)

##
print('\nPuanson characteristics:')
print('Мат ожидание: {:.5f}'.format(float(lamb)))
print('Дисперсия: {:.5f}'.format(float(lamb)))
print('Среднее квадратическое отклонение: {:.5f}'.format(float(math.sqrt(lamb))))
print('Мода: {:.5f}'.format(float(lamb)))
print('Медиана: {:.5f}'.format(float(lamb + 1/3 - 0.002/lamb)))
print('Коэф ассиметрии: {:.5f}'.format(float(1/math.sqrt(lamb))))
print('Коэф эксцесса : {:.5f}'.format(float((1/lamb))))
##
plt.show()
