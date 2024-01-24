import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.special as ss

x = pd.read_csv('data1.csv', encoding='utf-8', header=None)
x.columns = ['a', 'b']

x1 = x['b'][1: 64]
x2 = np.array(x1.values.tolist())
x3 = x2.astype(np.float64)

ys = x3[0:len(x3)-0]
ys1 = x3[len(x3)-0:len(x3)]

n, s, s1, rep = int(len(ys)), int(50000), int(10000), int(1)
y = np.zeros(shape=(rep, n))
for i in range(rep):
    y[i] = ys

n_1 = int(len(ys1))
y_1 = np.zeros(shape=(rep, n_1))
for i in range(rep):
    y_1[i] = ys1

lpd = np.zeros(rep)
lpdo = np.zeros(rep)
elp_11 = np.zeros(rep)


def f(y1, beta1, p11, p21, sigma1):
    q1 = np.zeros(n - 1)
    for k in range(n - 1):
        if y1[k + 1] <= (beta1[0] + y1[k] * beta1[1]):
            q1[k] = np.exp(-np.power((beta1[0] + y1[k] * beta1[1] - y1[k + 1]) * (
                    2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1[k] = np.exp(-np.power((y1[k + 1] - beta1[0] - beta1[1] * y1[k]) * (
                    2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
    return np.prod(q1)


beta_f = np.zeros(shape=(rep, s - s1, 2))
p1_f = np.zeros(shape=(rep, s - s1))
p2_f = np.zeros(shape=(rep, s - s1))
sigma_f = np.zeros(shape=(rep, s - s1))
wai = np.zeros(rep)

c1, d1 = float(3), float(2)
beta = np.zeros(shape=(rep, s, 2))
zeta = np.zeros(shape=(rep, s))
p1, p2, sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q2 in range(rep):
    zeta[q2][0] = 1 / 10
    for d in range(1, s):
        zeta[q2][d] = 1 / (10 * pow(d, 0.5))

mu_beta = np.zeros(shape=(rep, s, 2))
sigma_beta = np.zeros(shape=(rep, s, 2, 2))
mu_p1, sigma_p1 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_p2, sigma_p2 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_sigma, psi_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
u_sigma, v_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q in range(rep):
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0]), np.array([[5, 0], [0, 5]]), size=1)
    p1[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    p2[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0])
    sigma_beta[q][0] = np.array([[5, 0], [0, 5]])
    mu_p1[q][0], sigma_p1[q][0] = float(0), float(1)
    mu_p2[q][0], sigma_p2[q][0] = float(0), float(1)
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1]) /
                        f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1])) *
                  (st.multivariate_normal.pdf(ab1, np.array([0, 0]), np.array([[5, 0], [0, 5]])) /
                   st.multivariate_normal.pdf(beta[q][z - 1], np.array([0, 0]),
                                              np.array([[5, 0], [0, 5]]))) *
                  (st.multivariate_normal.pdf(beta[q][z - 1], mean=mu_beta[q][z - 1], cov=sigma_beta[q][z - 1]) /
                   st.multivariate_normal.pdf(ab1, mean=mu_beta[q][z - 1], cov=sigma_beta[q][z - 1])))
        if u1 < la1:
            beta[q][z] = ab1
        else:
            beta[q][z] = beta[q][z - 1]

        mu_beta[q][z] = mu_beta[q][z - 1] + zeta[q][z] * (beta[q][z] - mu_beta[q][z - 1])
        sigma_beta[q][z] = sigma_beta[q][z - 1] + zeta[q][z] * (
                np.dot(np.array([beta[q][z] - mu_beta[q][z - 1]]).T, np.array([beta[q][z] - mu_beta[q][z - 1]])) -
                sigma_beta[q][z - 1])

        ab2 = st.truncnorm.rvs(a=-mu_p1[q][z - 1] / pow(sigma_p1[q][z - 1], 0.5),
                               b=(5 - mu_p1[q][z - 1]) / pow(sigma_p1[q][z - 1], 0.5), loc=mu_p1[q][z - 1],
                               scale=pow(sigma_p1[q][z - 1], 0.5), size=1)
        u2 = st.uniform.rvs(loc=0, scale=1, size=1)
        la2 = min(1.0, (
                f(y[q], beta[q][z - 1], ab2, p2[q][z - 1], sigma[q][z - 1]) / f(y[q], beta[q][z - 1], p1[q][z - 1],
                                                                                p2[q][z - 1], sigma[q][z - 1])) * (
                          st.uniform.pdf(ab2, loc=0, scale=5) / st.uniform.pdf(p1[q][z - 1], loc=0, scale=5)) *
                  (st.truncnorm.pdf(p1[q][z - 1], a=-mu_p1[q][z - 1] / pow(sigma_p1[q][z - 1], 0.5),
                                    b=(5 - mu_p1[q][z - 1]) / pow(sigma_p1[q][z - 1], 0.5), loc=mu_p1[q][z - 1],
                                    scale=pow(sigma_p1[q][z - 1], 0.5)) /
                   st.truncnorm.pdf(ab2, a=-mu_p1[q][z - 1] / pow(sigma_p1[q][z - 1], 0.5),
                                    b=(5 - mu_p1[q][z - 1]) / pow(sigma_p1[q][z - 1], 0.5),
                                    loc=mu_p1[q][z - 1], scale=pow(sigma_p1[q][z - 1], 0.5))))
        if u2 < la2:
            p1[q][z] = ab2
        else:
            p1[q][z] = p1[q][z - 1]

        mu_p1[q][z] = mu_p1[q][z - 1] + zeta[q][z] * (p1[q][z] - mu_p1[q][z - 1])
        sigma_p1[q][z] = sigma_p1[q][z - 1] + zeta[q][z] * (pow((p1[q][z] - mu_p1[q][z - 1]), 2) - sigma_p1[q][z - 1])

        ab3 = st.truncnorm.rvs(a=-mu_p2[q][z - 1] / pow(sigma_p2[q][z - 1], 0.5),
                               b=(5 - mu_p2[q][z - 1]) / pow(sigma_p2[q][z - 1], 0.5), loc=mu_p2[q][z - 1],
                               scale=pow(sigma_p2[q][z - 1], 0.5), size=1)
        u3 = st.uniform.rvs(loc=0, scale=1, size=1)
        la3 = min(1.0, (
                f(y[q], beta[q][z - 1], p1[q][z - 1], ab3, sigma[q][z - 1]) / f(y[q], beta[q][z - 1], p1[q][z - 1],
                                                                                p2[q][z - 1], sigma[q][z - 1])) * (
                          st.uniform.pdf(ab3, loc=0, scale=5) / st.uniform.pdf(p2[q][z - 1], loc=0, scale=5)) *
                  (st.truncnorm.pdf(p2[q][z - 1], a=-mu_p2[q][z - 1] / pow(sigma_p2[q][z - 1], 0.5),
                                    b=(5 - mu_p2[q][z - 1]) / pow(sigma_p2[q][z - 1], 0.5), loc=mu_p2[q][z - 1],
                                    scale=pow(sigma_p2[q][z - 1], 0.5)) /
                   st.truncnorm.pdf(ab3, a=-mu_p2[q][z - 1] / pow(sigma_p2[q][z - 1], 0.5),
                                    b=(5 - mu_p2[q][z - 1]) / pow(sigma_p2[q][z - 1], 0.5), loc=mu_p2[q][z - 1],
                                    scale=pow(sigma_p2[q][z - 1], 0.5))))
        if u3 < la3:
            p2[q][z] = ab3
        else:
            p2[q][z] = p2[q][z - 1]

        mu_p2[q][z] = mu_p2[q][z - 1] + zeta[q][z] * (p2[q][z] - mu_p2[q][z - 1])
        sigma_p2[q][z] = sigma_p2[q][z - 1] + zeta[q][z] * (pow((p2[q][z] - mu_p2[q][z - 1]), 2) - sigma_p2[q][z - 1])

        ab4 = st.invgamma.rvs(a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1], size=1)
        u4 = st.uniform.rvs(loc=0, scale=1, size=1)
        la4 = min(1.0, (f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], ab4) / f(y[q], beta[q][z - 1], p1[q][z - 1],
                                                                                     p2[q][z - 1], sigma[q][z - 1])) *
                  (st.invgamma.pdf(ab4, a=c1, loc=0, scale=d1) / st.invgamma.pdf(sigma[q][z - 1], a=c1, loc=0,
                                                                                 scale=d1)) *
                  (st.invgamma.pdf(sigma[q][z - 1], a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1]) /
                   st.invgamma.pdf(ab4, a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1])))
        if u4 < la4:
            sigma[q][z] = ab4
        else:
            sigma[q][z] = sigma[q][z - 1]

        mu_sigma[q][z] = mu_sigma[q][z - 1] + zeta[q][z] * (sigma[q][z] - mu_sigma[q][z - 1])
        psi_sigma[q][z] = psi_sigma[q][z - 1] + zeta[q][z] * (
                pow(sigma[q][z] - mu_sigma[q][z - 1], 2) - psi_sigma[q][z - 1])

        u_sigma[q][z] = np.power(mu_sigma[q][z], 2) / psi_sigma[q][z] + 2
        v_sigma[q][z] = mu_sigma[q][z] * (np.power(mu_sigma[q][z], 2) / psi_sigma[q][z] + 1)

    beta_f[q] = beta[q][s1:s, :]
    p1_f[q] = p1[q][s1:s]
    p2_f[q] = p2[q][s1:s]
    sigma_f[q] = sigma[q][s1:s]


    def g(y1, k1, beta1, p11, p21, sigma1):
        if y1[k1 + 1] <= (beta1[0] + y1[k1] * beta1[1]):
            q1 = np.exp(-np.power((beta1[0] + y1[k1] * beta1[1] - y1[k1 + 1]) * (
                    2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1 = np.exp(-np.power((y1[k1 + 1] - beta1[0] - beta1[1] * y1[k1]) * (
                    2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
        return q1


    def g1(qs, jj):
        lpd_1 = np.zeros(s - s1)
        for r_1 in range(s - s1):
            lpd_1[r_1] = g(y[qs], jj, beta_f[qs][r_1], p1_f[qs][r_1], p2_f[qs][r_1], sigma_f[qs][r_1])
        return np.log(sum(lpd_1) / (s - s1))


    lpd2 = np.zeros(shape=(rep, n - 1))
    for r2 in range(n - 1):
        lpd2[q][r2] = g1(q, r2)
    lpd[q] = sum(lpd2[q])


    def g2(qq, jk):
        wai1 = np.zeros(s - s1)
        wai2 = np.zeros(s - s1)
        for n1 in range(s - s1):
            wai1[n1] = np.log(g(y[qq], jk, beta_f[qq][n1], p1_f[qq][n1], p2_f[qq][n1], sigma_f[qq][n1]))
        for n2 in range(s - s1):
            wai2[n2] = pow(wai1[n2] - np.mean(wai1), 2)
        return sum(wai2) / (s - s1 - 1)


    wai3 = np.zeros(shape=(rep, n - 1))
    for n3 in range(n - 1):
        wai3[q][n3] = g2(q, n3)
    wai[q] = -2 * (lpd[q] - sum(wai3[q]))


fi0 = np.zeros(shape=(rep, s - s1))
for r in range(rep):
    fi0[r] = beta[:, s1:s][r][:, 0]
fi_0 = fi0.reshape((s - s1) * rep, order='F')

fi1 = np.zeros(shape=(rep, s - s1))
for r in range(rep):
    fi1[r] = beta[:, s1:s][r][:, 1]
fi_1 = fi1.reshape((s - s1) * rep, order='F')

print('beta[0] %f' % np.mean(fi_0))
print('beta[1] %f' % np.mean(fi_1))
print('p1 %f' % np.mean(p1[:, s1:s]))
print('p2 %f' % np.mean(p2[:, s1:s]))
print('sigma %f' % np.mean(sigma[:, s1:s]))
print('WAIC %f' % wai)


def f(y1, beta1, sigma1):
    q1 = np.zeros(n - 1)
    for k in range(n - 1):
        if y1[k + 1] <= (beta1[0] + y1[k] * beta1[1]):
            q1[k] = np.exp(-(beta1[0] + y1[k] * beta1[1] - y1[k + 1]) * 2 / sigma1) / sigma1
        else:
            q1[k] = np.exp(-(y1[k + 1] - beta1[0] - beta1[1] * y1[k]) * 2 / sigma1) / sigma1
    return np.prod(q1)


beta_f = np.zeros(shape=(rep, s - s1, 2))
sigma_f = np.zeros(shape=(rep, s - s1))
wai = np.zeros(rep)

beta = np.zeros(shape=(rep, s, 2))
zeta = np.zeros(shape=(rep, s))
sigma = np.zeros(shape=(rep, s))

for q2 in range(rep):
    zeta[q2][0] = 1 / 10
    for d in range(1, s):
        zeta[q2][d] = 1 / (10 * pow(d, 0.5))

mu_beta = np.zeros(shape=(rep, s, 2))
sigma_beta = np.zeros(shape=(rep, s, 2, 2))
mu_sigma, psi_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
u_sigma, v_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q in range(rep):
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0]), np.array([[5, 0], [0, 5]]), size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0])
    sigma_beta[q][0] = np.array([[5, 0], [0, 5]])
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, sigma[q][z - 1]) / f(y[q], beta[q][z - 1], sigma[q][z - 1])) *
                  (st.multivariate_normal.pdf(ab1, np.array([0, 0]), np.array([[5, 0], [0, 5]])) /
                   st.multivariate_normal.pdf(beta[q][z - 1], np.array([0, 0]), np.array([[5, 0], [0, 5]]))) *
                  (st.multivariate_normal.pdf(beta[q][z - 1], mean=mu_beta[q][z - 1], cov=sigma_beta[q][z - 1]) /
                   st.multivariate_normal.pdf(ab1, mean=mu_beta[q][z - 1], cov=sigma_beta[q][z - 1])))
        if u1 < la1:
            beta[q][z] = ab1
        else:
            beta[q][z] = beta[q][z - 1]

        mu_beta[q][z] = mu_beta[q][z - 1] + zeta[q][z] * (beta[q][z] - mu_beta[q][z - 1])
        sigma_beta[q][z] = sigma_beta[q][z - 1] + zeta[q][z] * (
                np.dot(np.array([beta[q][z] - mu_beta[q][z - 1]]).T, np.array([beta[q][z] - mu_beta[q][z - 1]])) -
                sigma_beta[q][z - 1])

        ab4 = st.invgamma.rvs(a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1], size=1)
        u4 = st.uniform.rvs(loc=0, scale=1, size=1)
        la4 = min(1.0, (f(y[q], beta[q][z - 1], ab4) / f(y[q], beta[q][z - 1], sigma[q][z - 1])) *
                  (st.invgamma.pdf(ab4, a=c1, loc=0, scale=d1) / st.invgamma.pdf(sigma[q][z - 1], a=c1, loc=0,
                                                                                 scale=d1)) *
                  (st.invgamma.pdf(sigma[q][z - 1], a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1]) /
                   st.invgamma.pdf(ab4, a=u_sigma[q][z - 1], loc=0, scale=v_sigma[q][z - 1])))
        if u4 < la4:
            sigma[q][z] = ab4
        else:
            sigma[q][z] = sigma[q][z - 1]

        mu_sigma[q][z] = mu_sigma[q][z - 1] + zeta[q][z] * (sigma[q][z] - mu_sigma[q][z - 1])
        psi_sigma[q][z] = psi_sigma[q][z - 1] + zeta[q][z] * (
                pow(sigma[q][z] - mu_sigma[q][z - 1], 2) - psi_sigma[q][z - 1])

        u_sigma[q][z] = np.power(mu_sigma[q][z], 2) / psi_sigma[q][z] + 2
        v_sigma[q][z] = mu_sigma[q][z] * (np.power(mu_sigma[q][z], 2) / psi_sigma[q][z] + 1)

    beta_f[q] = beta[q][s1:s, :]
    sigma_f[q] = sigma[q][s1:s]


    def g(y1, k1, beta1, sigma1):
        if y1[k1 + 1] <= (beta1[0] + y1[k1] * beta1[1]):
            q1 = np.exp(-(beta1[0] + y1[k1] * beta1[1] - y1[k1 + 1]) * 2 / sigma1) / sigma1
        else:
            q1 = np.exp(-(y1[k1 + 1] - beta1[0] - beta1[1] * y1[k1]) * 2 / sigma1) / sigma1
        return q1


    def g1(qs, jj):
        lpd_1 = np.zeros(s - s1)
        for r_1 in range(s - s1):
            lpd_1[r_1] = g(y[qs], jj, beta_f[qs][r_1], sigma_f[qs][r_1])
        return np.log(sum(lpd_1) / (s - s1))


    lpd2 = np.zeros(shape=(rep, n - 1))
    for r2 in range(n - 1):
        lpd2[q][r2] = g1(q, r2)
    lpd[q] = sum(lpd2[q])


    def g2(qq, jk):
        wai1 = np.zeros(s - s1)
        wai2 = np.zeros(s - s1)
        for n1 in range(s - s1):
            wai1[n1] = np.log(g(y[qq], jk, beta_f[qq][n1], sigma_f[qq][n1]))
        for n2 in range(s - s1):
            wai2[n2] = pow(wai1[n2] - np.mean(wai1), 2)
        return sum(wai2) / (s - s1 - 1)


    wai3 = np.zeros(shape=(rep, n - 1))
    for n3 in range(n - 1):
        wai3[q][n3] = g2(q, n3)
    wai[q] = -2 * (lpd[q] - sum(wai3[q]))


fi0 = np.zeros(shape=(rep, s - s1))
for r in range(rep):
    fi0[r] = beta[:, s1:s][r][:, 0]
fi_0 = fi0.reshape((s - s1) * rep, order='F')

fi1 = np.zeros(shape=(rep, s - s1))
for r in range(rep):
    fi1[r] = beta[:, s1:s][r][:, 1]
fi_1 = fi1.reshape((s - s1) * rep, order='F')

print('beta[0] %f' % np.mean(fi_0))
print('beta[1] %f' % np.mean(fi_1))
print('sigma %f' % np.mean(sigma[:, s1:s]))
print('WAIC %f' % wai)
