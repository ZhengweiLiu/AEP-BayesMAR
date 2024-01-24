import numpy as np
import scipy.stats as st
import scipy.special as ss
import time

start = time.time()
n, n_1, s, s1, rep = int(100), int(1), int(50000), int(10000), int(25)
gen = int(10000)
beta_true = np.array([0.3, 0.75, -0.35])
p1_t, p2_t, sigma_t = float(1.0), float(0.5), float(1.0)
y_1 = np.zeros(shape=(rep, n + n_1))
y = np.zeros(shape=(rep, n))
y_2 = np.zeros(shape=(rep, n_1))


def ran_aep(p_1, p_2, sigma_1):
    u = st.bernoulli.rvs(p=0.5, loc=0, size=1)
    if u == 0:
        u_1 = st.gamma.rvs(a=1 + 1 / p_1, loc=0, scale=1, size=1)
        epi = st.uniform.rvs(loc=-sigma_1 * pow(u_1, 1 / p_1) / (2 * ss.gamma(1 + 1 / p_1)),
                             scale=sigma_1 * pow(u_1, 1 / p_1) / (2 * ss.gamma(1 + 1 / p_1)), size=1)
    else:
        u_2 = st.gamma.rvs(a=1 + 1 / p_2, loc=0, scale=1, size=1)
        epi = st.uniform.rvs(loc=0, scale=sigma_1 * pow(u_2, 1 / p_2) / (2 * ss.gamma(1 + 1 / p_2)), size=1)
    return epi


for e in range(rep):
    for i in range(2, n + n_1):
        y_1[e][0] = st.norm.rvs(size=1)
        y_1[e][1] = st.norm.rvs(size=1)
        y_1[e][i] = beta_true[0] + y_1[e][i - 1] * beta_true[1] + y_1[e][i - 2] * beta_true[2] + ran_aep(p1_t, p2_t,
                                                                                                         sigma_t)
for e1 in range(rep):
    y[e1] = y_1[e1][:n]

for e2 in range(rep):
    y_2[e2] = y_1[e2][n:]


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
wai_1 = np.zeros(rep)
lpd = np.zeros(rep)

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
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0]), np.array([[1, 0], [0, 1]]), size=1)
    p1[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    p2[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0])
    sigma_beta[q][0] = np.array([[1, 0], [0, 1]])
    mu_p1[q][0], sigma_p1[q][0] = float(0), float(1)
    mu_p2[q][0], sigma_p2[q][0] = float(0), float(1)
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1]) /
                        f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1])) *
                  (st.multivariate_normal.pdf(ab1, np.array([0, 0]), np.array([[1, 0], [0, 1]])) /
                   st.multivariate_normal.pdf(beta[q][z - 1], np.array([0, 0]),
                                              np.array([[1, 0], [0, 1]]))) *
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
    wai_1[q] = -2 * (lpd[q] - sum(wai3[q]))

fi0_mse_1 = np.zeros(rep)
for r in range(rep):
    fi0_mse_1[r] = np.mean(beta[:, s1:s][r][:, 0])
fi1_mse_1 = np.zeros(rep)
for r in range(rep):
    fi1_mse_1[r] = np.mean(beta[:, s1:s][r][:, 1])

gi3_mse_1 = np.zeros(rep)
for r in range(rep):
    gi3_mse_1[r] = np.mean(p1[:, s1:s][r])
gi4_mse_1 = np.zeros(rep)
for r in range(rep):
    gi4_mse_1[r] = np.mean(p2[:, s1:s][r])
gi5_mse_1 = np.zeros(rep)
for r in range(rep):
    gi5_mse_1[r] = np.mean(sigma[:, s1:s][r])


def f(y1, beta1, p11, p21, sigma1):
    q1 = np.zeros(n - 2)
    for k in range(n - 2):
        if y1[k + 2] <= (beta1[0] + y1[k + 1] * beta1[1] + y1[k] * beta1[2]):
            q1[k] = np.exp(-np.power((beta1[0] + y1[k + 1] * beta1[1] + beta1[2] * y1[k] - y1[k + 2]) * (
                    2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1[k] = np.exp(-np.power((y1[k + 2] - beta1[0] - beta1[1] * y1[k + 1] - beta1[2] * y1[k]) * (
                    2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
    return np.prod(q1)


beta_f = np.zeros(shape=(rep, s - s1, 3))
p1_f = np.zeros(shape=(rep, s - s1))
p2_f = np.zeros(shape=(rep, s - s1))
sigma_f = np.zeros(shape=(rep, s - s1))
wai_2 = np.zeros(rep)
lpd = np.zeros(rep)

c1, d1 = float(3), float(2)
beta = np.zeros(shape=(rep, s, 3))
zeta = np.zeros(shape=(rep, s))
p1, p2, sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q2 in range(rep):
    zeta[q2][0] = 1 / 10
    for d in range(1, s):
        zeta[q2][d] = 1 / (10 * pow(d, 0.5))

mu_beta = np.zeros(shape=(rep, s, 3))
sigma_beta = np.zeros(shape=(rep, s, 3, 3))
mu_p1, sigma_p1 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_p2, sigma_p2 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_sigma, psi_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
u_sigma, v_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q in range(rep):
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), size=1)
    p1[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    p2[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0, 0])
    sigma_beta[q][0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    mu_p1[q][0], sigma_p1[q][0] = float(0), float(1)
    mu_p2[q][0], sigma_p2[q][0] = float(0), float(1)
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1]) /
                        f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1])) *
                  (st.multivariate_normal.pdf(ab1, np.array([0, 0, 0]), np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])) /
                   st.multivariate_normal.pdf(beta[q][z - 1], np.array([0, 0, 0]),
                                              np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))) *
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
        if y1[k1 + 2] <= (beta1[0] + y1[k1 + 1] * beta1[1] + y1[k1] * beta1[2]):
            q1 = np.exp(-np.power((beta1[0] + y1[k1 + 1] * beta1[1] + beta1[2] * y1[k1] - y1[k1 + 2]) * (
                    2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1 = np.exp(-np.power((y1[k1 + 2] - beta1[0] - beta1[1] * y1[k1 + 1] - beta1[2] * y1[k1]) * (
                    2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
        return q1


    def g1(qs, jj):
        lpd_1 = np.zeros(s - s1)
        for r_1 in range(s - s1):
            lpd_1[r_1] = g(y[qs], jj, beta_f[qs][r_1], p1_f[qs][r_1], p2_f[qs][r_1], sigma_f[qs][r_1])
        return np.log(sum(lpd_1) / (s - s1))


    lpd2 = np.zeros(shape=(rep, n - 2))
    for r2 in range(n - 2):
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


    wai3 = np.zeros(shape=(rep, n - 2))
    for n3 in range(n - 2):
        wai3[q][n3] = g2(q, n3)
    wai_2[q] = -2 * (lpd[q] - sum(wai3[q]))

fi0_mse_2 = np.zeros(rep)
for r in range(rep):
    fi0_mse_2[r] = np.mean(beta[:, s1:s][r][:, 0])
fi1_mse_2 = np.zeros(rep)
for r in range(rep):
    fi1_mse_2[r] = np.mean(beta[:, s1:s][r][:, 1])
fi2_mse_2 = np.zeros(rep)
for r in range(rep):
    fi2_mse_2[r] = np.mean(beta[:, s1:s][r][:, 2])

gi3_mse_2 = np.zeros(rep)
for r in range(rep):
    gi3_mse_2[r] = np.mean(p1[:, s1:s][r])
gi4_mse_2 = np.zeros(rep)
for r in range(rep):
    gi4_mse_2[r] = np.mean(p2[:, s1:s][r])
gi5_mse_2 = np.zeros(rep)
for r in range(rep):
    gi5_mse_2[r] = np.mean(sigma[:, s1:s][r])


def f(y1, beta1, p11, p21, sigma1):
    q1 = np.zeros(n - 3)
    for k in range(n - 3):
        if y1[k + 3] <= (beta1[0] + y1[k + 2] * beta1[1] + y1[k + 1] * beta1[2] + y1[k] * beta1[3]):
            q1[k] = np.exp(
                -np.power((beta1[0] + y1[k + 2] * beta1[1] + y1[k + 1] * beta1[2] + y1[k] * beta1[3] - y1[k + 3]) * (
                        2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1[k] = np.exp(
                -np.power((y1[k + 3] - beta1[0] - y1[k + 2] * beta1[1] - y1[k + 1] * beta1[2] - y1[k] * beta1[3]) * (
                        2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
    return np.prod(q1)


beta_f = np.zeros(shape=(rep, s - s1, 4))
p1_f = np.zeros(shape=(rep, s - s1))
p2_f = np.zeros(shape=(rep, s - s1))
sigma_f = np.zeros(shape=(rep, s - s1))
wai_3 = np.zeros(rep)

c1, d1 = float(3), float(2)
beta = np.zeros(shape=(rep, s, 4))
zeta = np.zeros(shape=(rep, s))
p1, p2, sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q2 in range(rep):
    zeta[q2][0] = 1 / 10
    for d in range(1, s):
        zeta[q2][d] = 1 / (10 * pow(d, 0.5))

mu_beta = np.zeros(shape=(rep, s, 4))
sigma_beta = np.zeros(shape=(rep, s, 4, 4))
mu_p1, sigma_p1 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_p2, sigma_p2 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_sigma, psi_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
u_sigma, v_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q in range(rep):
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0, 0, 0]), np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                                                                              [0, 0, 1, 0], [0, 0, 0, 1]]), size=1)
    p1[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    p2[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0, 0, 0])
    sigma_beta[q][0] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    mu_p1[q][0], sigma_p1[q][0] = float(0), float(1)
    mu_p2[q][0], sigma_p2[q][0] = float(0), float(1)
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1]) /
                        f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1])) *
                  (st.multivariate_normal.pdf(ab1, np.array([0, 0, 0, 0]),
                                              np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])) /
                   st.multivariate_normal.pdf(beta[q][z - 1], np.array([0, 0, 0, 0]),
                                              np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))) *
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
        if y1[k1 + 3] <= (beta1[0] + y1[k1 + 2] * beta1[1] + y1[k1 + 1] * beta1[2] + y1[k1] * beta1[3]):
            q1 = np.exp(
                -np.power(
                    (beta1[0] + y1[k1 + 2] * beta1[1] + y1[k1 + 1] * beta1[2] + y1[k1] * beta1[3] - y1[k1 + 3]) * (
                            2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1 = np.exp(
                -np.power(
                    (y1[k1 + 3] - beta1[0] - y1[k1 + 2] * beta1[1] - y1[k1 + 1] * beta1[2] - y1[k1] * beta1[3]) * (
                            2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
        return q1


    def g1(qs, jj):
        lpd_1 = np.zeros(s - s1)
        for r_1 in range(s - s1):
            lpd_1[r_1] = g(y[qs], jj, beta_f[qs][r_1], p1_f[qs][r_1], p2_f[qs][r_1], sigma_f[qs][r_1])
        return np.log(sum(lpd_1) / (s - s1))


    lpd2 = np.zeros(shape=(rep, n - 3))
    for r2 in range(n - 3):
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


    wai3 = np.zeros(shape=(rep, n - 3))
    for n3 in range(n - 3):
        wai3[q][n3] = g2(q, n3)
    wai_3[q] = -2 * (lpd[q] - sum(wai3[q]))

fi0_mse_3 = np.zeros(rep)
for r in range(rep):
    fi0_mse_3[r] = np.mean(beta[:, s1:s][r][:, 0])
fi1_mse_3 = np.zeros(rep)
for r in range(rep):
    fi1_mse_3[r] = np.mean(beta[:, s1:s][r][:, 1])
fi2_mse_3 = np.zeros(rep)
for r in range(rep):
    fi2_mse_3[r] = np.mean(beta[:, s1:s][r][:, 2])
fi3_mse_3 = np.zeros(rep)
for r in range(rep):
    fi3_mse_3[r] = np.mean(beta[:, s1:s][r][:, 3])

gi3_mse_3 = np.zeros(rep)
for r in range(rep):
    gi3_mse_3[r] = np.mean(p1[:, s1:s][r])
gi4_mse_3 = np.zeros(rep)
for r in range(rep):
    gi4_mse_3[r] = np.mean(p2[:, s1:s][r])
gi5_mse_3 = np.zeros(rep)
for r in range(rep):
    gi5_mse_3[r] = np.mean(sigma[:, s1:s][r])


wai_final_1 = np.zeros(shape=(3, rep))
wai_final_1[0] = - wai_1 / 2
wai_final_1[1] = - wai_2 / 2
wai_final_1[2] = - wai_3 / 2

wai_final_2 = np.transpose(wai_final_1)

weight = np.zeros(shape=(rep, 3))
for i in range(rep):
    for j in range(3):
        weight[i][j] = np.exp(wai_final_2[i][j]) / (
                np.exp(wai_final_2[i][0]) + np.exp(wai_final_2[i][1]) + np.exp(wai_final_2[i][2]))


def step_one(pc1, be1, p1_1, p2_1, sig_1):
    o1 = int(len(be1) - 1)
    sc1 = np.append(1, pc1[n - o1:n])
    return np.dot(sc1, be1) + ran_aep(p1_1, p2_1, sig_1)


def step_two(pc2, be2, p1_2, p2_2, sig_2):
    o2 = int(len(be2) - 1)
    cc1 = pc2[n - o2:n]
    s_m1 = np.dot(np.append(1, cc1), be2) + ran_aep(p1_2, p2_2, sig_2)
    cc2 = np.append(cc1[1:len(cc1)], s_m1)
    s_m2 = np.dot(np.append(1, cc2), be2) + ran_aep(p1_2, p2_2, sig_2)
    return np.append(s_m1, s_m2)


def step_three(pc3, be3, p1_3, p2_3, sig_3):
    o3 = int(len(be3) - 1)
    cs1 = pc3[n - o3:n]
    s_k1 = np.dot(np.append(1, cs1), be3) + ran_aep(p1_3, p2_3, sig_3)
    cs2 = np.append(cs1[1:len(cs1)], s_k1)
    s_k2 = np.dot(np.append(1, cs2), be3) + ran_aep(p1_3, p2_3, sig_3)
    cs3 = np.append(cs2[1:len(cs2)], s_k2)
    s_k3 = np.dot(np.append(1, cs3), be3) + ran_aep(p1_3, p2_3, sig_3)
    return np.append(np.append(s_k1, s_k2), s_k3)


def step_four(pc4, be4, p1_4, p2_4, sig_4):
    o4 = int(len(be4) - 1)
    cw1 = pc4[n - o4:n]
    s_w1 = np.dot(np.append(1, cw1), be4) + ran_aep(p1_4, p2_4, sig_4)
    cw2 = np.append(cw1[1:len(cw1)], s_w1)
    s_w2 = np.dot(np.append(1, cw2), be4) + ran_aep(p1_4, p2_4, sig_4)
    cw3 = np.append(cw2[1:len(cw2)], s_w2)
    s_w3 = np.dot(np.append(1, cw3), be4) + ran_aep(p1_4, p2_4, sig_4)
    cw4 = np.append(cw3[1:len(cw3)], s_w3)
    s_w4 = np.dot(np.append(1, cw4), be4) + ran_aep(p1_4, p2_4, sig_4)
    return np.append(np.append(np.append(s_w1, s_w2), s_w3), s_w4)


out_true = np.zeros(shape=(rep, gen, n_1))
for i in range(rep):
    for j in range(gen):
        out_true[i][j] = step_one(y[i], [fi0_mse_2[i], fi1_mse_2[i], fi2_mse_2[i]], gi3_mse_2[i],
                                  gi4_mse_2[i], gi5_mse_2[i])

out_BMA = np.zeros(shape=(rep, gen, n_1))
for i in range(rep):
    for j in range(gen):
        out_BMA[i][j] = weight[i][0] * step_one(y[i], [fi0_mse_1[i], fi1_mse_1[i]], gi3_mse_1[i], gi4_mse_1[i],
                                                gi5_mse_1[i]) + \
                        weight[i][1] * step_one(y[i], [fi0_mse_2[i], fi1_mse_2[i], fi2_mse_2[i]], gi3_mse_2[i],
                                                gi4_mse_2[i], gi5_mse_2[i]) + \
                        weight[i][2] * step_one(y[i], [fi0_mse_3[i], fi1_mse_3[i], fi2_mse_3[i], fi3_mse_3[i]],
                                                gi3_mse_3[i], gi4_mse_3[i], gi5_mse_3[i])

out_true_1 = np.zeros(shape=(rep, gen))
for i in range(rep):
    for j in range(gen):
        out_true_1[i][j] = np.sum(abs(out_true[i][j] - y_2[i])) / gen

out_BMA_1 = np.zeros(shape=(rep, gen))
for i in range(rep):
    for j in range(gen):
        out_BMA_1[i][j] = np.sum(abs(out_BMA[i][j] - y_2[i])) / gen

out_true_final = np.sum(out_true_1, axis=1)
out_BMA_final = np.sum(out_BMA_1, axis=1)

out_true_final_1 = np.sum(out_true_final) / rep
out_BMA_final_1 = np.sum(out_BMA_final) / rep

print(out_true_final_1)
print(out_BMA_final_1)

end = time.time()
print('Running time %f' % (end - start))
