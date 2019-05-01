# -*- coding: utf-8 -*-
# Author: chenling
# Created at: 04/05/19 10:52 AM

import data_helper as dh
import numpy as np
from matplotlib import pyplot as plt
from hawkeslib.model.uv_exp import UnivariateExpHawkesProcess as UVHP

df = dh.load_json("2.json")

# take all mark 21 and fit a univariate exp HP
#td = np.array(df.loc[df.mark == 21, :].get("time")).astype(np.float64)

time_list = [dh.parse_datetime(i) for i in list(df.created_at)][10203:12775][::-1]

new_time_list = [(time_list[i] - time_list[0]).total_seconds()*0.01 for i in range(len(time_list))]

new_time_list = np.array(new_time_list).astype(np.float64)

proc = UVHP()
proc.fit(new_time_list, method="em")

mu, alpha, theta = proc.get_params()

lda_ar = [mu + np.sum(alpha * theta * np.exp(-theta * (x - new_time_list[new_time_list < x]))) \
          for x in np.arange(0, 26430, 20)]

plt.figure(figsize=(12,3))
plt.plot(np.arange(0, 26430, 20), lda_ar, 'g-')
ax = plt.gca()
plt.axvline(x=15100, c='r', label='Snopes involving time')
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = '06-05'
labels[2] = '06-10'
labels[3] = '06-15'
labels[4] = '06-20'
labels[5] = '06-25'
labels[6] = '06-30'
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.set_tick_params(labelsize=13)
plt.ylabel("$\lambda(t)$", fontsize=20)
plt.xlabel("$t$", fontsize=20)
plt.legend(fontsize=14)
plt.tight_layout()
plt.savefig('intensity_false.png', dpi=500)



df_2 = dh.load_json("59.json")

time_list = sorted([dh.parse_datetime(i) for i in list(df_2.created_at)])[916:1648]

new_time_list = [(time_list[i] - time_list[0]).total_seconds()*0.01 for i in range(len(time_list))]

new_time_list = np.array(new_time_list).astype(np.float64)

proc = UVHP()
proc.fit(new_time_list, method="em")

mu, alpha, theta = proc.get_params()

lda_ar = [mu + np.sum(alpha * theta * np.exp(-theta * (x - new_time_list[new_time_list < x]))) \
          for x in np.arange(0, 26640, 20)]

plt.figure(figsize=(10,3))
plt.plot(np.arange(0, 26640, 20), lda_ar, 'g-')
ax = plt.gca()
plt.axvline(x=4900, c='r', label='Snopes involving time')
labels = [item.get_text() for item in ax.get_xticklabels()]
labels[1] = '07-05'
labels[2] = '07-10'
labels[3] = '07-15'
labels[4] = '07-20'
labels[5] = '07-25'
labels[6] = '07-30'
ax.set_xticklabels(labels, fontsize=15)
ax.yaxis.set_tick_params(labelsize=13)
plt.ylabel("$\lambda^(t)$", fontsize=20)
plt.xlabel("$t$", fontsize=20)
plt.legend(fontsize=14)
plt.tight_layout()
plt.show()
#plt.savefig('intensity_true.png', dpi=500)
