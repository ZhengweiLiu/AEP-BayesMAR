import numpy as np
import scipy.stats as st
import scipy.special as ss
import time

start = time.time()
n, s, s1, rep = int(100), int(50000), int(10000), int(100)
beta_true = np.array([0.2, 0.4, 0, 0, 0.3, 0, 0, 0, 0.2])
p1_t, p2_t, sigma_t = float(1.0), float(2.0), float(2.0)
y = np.zeros(shape=(rep, n))


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
    for i in range(8, n):
        y[e][0] = st.norm.rvs(size=1)
        y[e][1] = st.norm.rvs(size=1)
        y[e][2] = st.norm.rvs(size=1)
        y[e][3] = st.norm.rvs(size=1)
        y[e][4] = st.norm.rvs(size=1)
        y[e][5] = st.norm.rvs(size=1)
        y[e][6] = st.norm.rvs(size=1)
        y[e][7] = st.norm.rvs(size=1)
        y[e][i] = beta_true[0] + y[e][i - 1] * beta_true[1] + y[e][i - 2] * beta_true[2] + y[e][i - 3] * beta_true[3] + \
                  y[e][i - 4] * beta_true[4] + y[e][i - 5] * beta_true[5] + y[e][i - 6] * beta_true[6] + y[e][i - 7] * \
                  beta_true[7] + y[e][i - 8] * beta_true[8] + ran_aep(p1_t, p2_t, sigma_t)


def f(y1, beta1, p11, p21, sigma1):
    q1 = np.zeros(n - 8)
    for k in range(n - 8):
        if y1[k + 8] <= (
                beta1[0] + y1[k + 7] * beta1[1] + y1[k + 6] * beta1[2] + y1[k + 5] * beta1[3] + y1[k + 4] * beta1[4] +
                y1[k + 3] * beta1[5] + y1[k + 2] * beta1[6] + y1[k + 1] * beta1[7] + y1[k] * beta1[8]):
            q1[k] = np.exp(-np.power(
                (beta1[0] + y1[k + 7] * beta1[1] + y1[k + 6] * beta1[2] + y1[k + 5] * beta1[3] + y1[k + 4] * beta1[4] +
                 y1[k + 3] * beta1[5] + y1[k + 2] * beta1[6] + y1[k + 1] * beta1[7] + y1[k] * beta1[8] - y1[k + 8]) * (
                        2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1[k] = np.exp(-np.power((y1[k + 8] - beta1[0] - y1[k + 7] * beta1[1] - y1[k + 6] * beta1[2] - y1[
                k + 5] * beta1[3] - y1[k + 4] * beta1[4] - y1[k + 3] * beta1[5] - y1[k + 2] * beta1[6] - y1[k + 1] *
                                      beta1[7] - y1[k] * beta1[8]) * (
                                             2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
    return np.prod(q1)


beta_f = np.zeros(shape=(rep, s - s1, 9))
p1_f = np.zeros(shape=(rep, s - s1))
p2_f = np.zeros(shape=(rep, s - s1))
sigma_f = np.zeros(shape=(rep, s - s1))
wai = np.zeros(rep)
lpd = np.zeros(rep)

c1, d1 = float(3), float(2)
beta = np.zeros(shape=(rep, s, 9))
zeta = np.zeros(shape=(rep, s))
p1, p2, sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q2 in range(rep):
    zeta[q2][0] = 1 / 10
    for d in range(1, s):
        zeta[q2][d] = 1 / (10 * pow(d, 0.5))

mu_beta = np.zeros(shape=(rep, s, 9))
sigma_beta = np.zeros(shape=(rep, s, 9, 9))
mu_p1, sigma_p1 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_p2, sigma_p2 = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
mu_sigma, psi_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))
u_sigma, v_sigma = np.zeros(shape=(rep, s)), np.zeros(shape=(rep, s))

for q in range(rep):
    beta[q][0] = st.multivariate_normal.rvs(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                            np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                      [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                      [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                      [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                      [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                      [0, 0, 0, 0, 0, 0, 0, 0, 1]]), size=1)
    p1[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    p2[q][0] = st.uniform.rvs(loc=0, scale=5, size=1)
    sigma[q][0] = st.invgamma.rvs(a=c1, loc=0, scale=d1, size=1)

    mu_beta[q][0] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
    sigma_beta[q][0] = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 1]])
    mu_p1[q][0], sigma_p1[q][0] = float(0), float(1)
    mu_p2[q][0], sigma_p2[q][0] = float(0), float(1)
    mu_sigma[q][0], psi_sigma[q][0] = float(0), float(1)
    u_sigma[q][0], v_sigma[q][0] = float(3), float(2)

    for z in range(1, s):
        ab1 = st.multivariate_normal.rvs(mu_beta[q][z - 1], sigma_beta[q][z - 1], size=1)
        u1 = st.uniform.rvs(loc=0, scale=1, size=1)
        la1 = min(1.0, (f(y[q], ab1, p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1]) /
                        f(y[q], beta[q][z - 1], p1[q][z - 1], p2[q][z - 1], sigma[q][z - 1])) * (
                          st.multivariate_normal.pdf(ab1, mean=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                                     cov=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 1]])) /
                          st.multivariate_normal.pdf(beta[q][z - 1], mean=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                                     cov=np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 1, 0, 0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 1, 0, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 1, 0, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 1, 0, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 1, 0, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 1, 0, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 1, 0],
                                                                   [0, 0, 0, 0, 0, 0, 0, 0, 1]]))) *
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
        if y1[k1 + 8] <= (
                beta1[0] + y1[k1 + 7] * beta1[1] + y1[k1 + 6] * beta1[2] + y1[k1 + 5] * beta1[3] + y1[k1 + 4] *
                beta1[4] + y1[k1 + 3] * beta1[5] + y1[k1 + 2] * beta1[6] + y1[k1 + 1] * beta1[7] + y1[k1] * beta1[8]):
            q1 = np.exp(-np.power(
                (beta1[0] + y1[k1 + 7] * beta1[1] + y1[k1 + 6] * beta1[2] + y1[k1 + 5] * beta1[3] + y1[k1 + 4] *
                 beta1[4] + y1[k1 + 3] * beta1[5] + y1[k1 + 2] * beta1[6] + y1[k1 + 1] * beta1[7] + y1[k1] * beta1[8] -
                 y1[k1 + 8]) * (2 * ss.gamma(1 + 1 / p11)) / sigma1, p11)) / sigma1
        else:
            q1 = np.exp(-np.power((y1[k1 + 8] - beta1[0] - y1[k1 + 7] * beta1[1] - y1[k1 + 6] * beta1[2] - y1[
                k1 + 5] * beta1[3] - y1[k1 + 4] * beta1[4] - y1[k1 + 3] * beta1[5] - y1[k1 + 2] * beta1[6] - y1[
                                       k1 + 1] * beta1[7] - y1[k1] * beta1[8]) * (
                                          2 * ss.gamma(1 + 1 / p21)) / sigma1, p21)) / sigma1
        return q1


    def g1(qs, jj):
        lpd_1 = np.zeros(s - s1)
        for r_1 in range(s - s1):
            lpd_1[r_1] = g(y[qs], jj, beta_f[qs][r_1], p1_f[qs][r_1], p2_f[qs][r_1], sigma_f[qs][r_1])
        return np.log(sum(lpd_1) / (s - s1))


    lpd2 = np.zeros(shape=(rep, n - 8))
    for r2 in range(n - 8):
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


    wai3 = np.zeros(shape=(rep, n - 8))
    for n3 in range(n - 8):
        wai3[q][n3] = g2(q, n3)
    wai[q] = -2 * (lpd[q] - sum(wai3[q]))

fi0_mse = np.zeros(rep)
for r in range(rep):
    fi0_mse[r] = np.mean(beta[:, s1:s][r][:, 0])
fi0_root_mse = np.zeros(rep)
for r in range(rep):
    fi0_root_mse[r] = pow(fi0_mse[r] - beta_true[0], 2)
fi0_final = pow(np.sum(fi0_root_mse) / rep, 0.5)

fi1_mse = np.zeros(rep)
for r in range(rep):
    fi1_mse[r] = np.mean(beta[:, s1:s][r][:, 1])
fi1_root_mse = np.zeros(rep)
for r in range(rep):
    fi1_root_mse[r] = pow(fi1_mse[r] - beta_true[1], 2)
fi1_final = pow(np.sum(fi1_root_mse) / rep, 0.5)

fi2_mse = np.zeros(rep)
for r in range(rep):
    fi2_mse[r] = np.mean(beta[:, s1:s][r][:, 2])
fi2_root_mse = np.zeros(rep)
for r in range(rep):
    fi2_root_mse[r] = pow(fi2_mse[r] - beta_true[2], 2)
fi2_final = pow(np.sum(fi2_root_mse) / rep, 0.5)

fi3_mse = np.zeros(rep)
for r in range(rep):
    fi3_mse[r] = np.mean(beta[:, s1:s][r][:, 3])
fi3_root_mse = np.zeros(rep)
for r in range(rep):
    fi3_root_mse[r] = pow(fi3_mse[r] - beta_true[3], 2)
fi3_final = pow(np.sum(fi3_root_mse) / rep, 0.5)

fi4_mse = np.zeros(rep)
for r in range(rep):
    fi4_mse[r] = np.mean(beta[:, s1:s][r][:, 4])
fi4_root_mse = np.zeros(rep)
for r in range(rep):
    fi4_root_mse[r] = pow(fi4_mse[r] - beta_true[4], 2)
fi4_final = pow(np.sum(fi4_root_mse) / rep, 0.5)

fi5_mse = np.zeros(rep)
for r in range(rep):
    fi5_mse[r] = np.mean(beta[:, s1:s][r][:, 5])
fi5_root_mse = np.zeros(rep)
for r in range(rep):
    fi5_root_mse[r] = pow(fi5_mse[r] - beta_true[5], 2)
fi5_final = pow(np.sum(fi5_root_mse) / rep, 0.5)

fi6_mse = np.zeros(rep)
for r in range(rep):
    fi6_mse[r] = np.mean(beta[:, s1:s][r][:, 6])
fi6_root_mse = np.zeros(rep)
for r in range(rep):
    fi6_root_mse[r] = pow(fi6_mse[r] - beta_true[6], 2)
fi6_final = pow(np.sum(fi6_root_mse) / rep, 0.5)

fi7_mse = np.zeros(rep)
for r in range(rep):
    fi7_mse[r] = np.mean(beta[:, s1:s][r][:, 7])
fi7_root_mse = np.zeros(rep)
for r in range(rep):
    fi7_root_mse[r] = pow(fi7_mse[r] - beta_true[7], 2)
fi7_final = pow(np.sum(fi7_root_mse) / rep, 0.5)

fi8_mse = np.zeros(rep)
for r in range(rep):
    fi8_mse[r] = np.mean(beta[:, s1:s][r][:, 8])
fi8_root_mse = np.zeros(rep)
for r in range(rep):
    fi8_root_mse[r] = pow(fi8_mse[r] - beta_true[8], 2)
fi8_final = pow(np.sum(fi8_root_mse) / rep, 0.5)

gi3_mse = np.zeros(rep)
for r in range(rep):
    gi3_mse[r] = np.mean(p1[:, s1:s][r])
gi3_root_mse = np.zeros(rep)
for r in range(rep):
    gi3_root_mse[r] = pow(gi3_mse[r] - p1_t, 2)
gi3_final = pow(np.sum(gi3_root_mse) / rep, 0.5)

gi4_mse = np.zeros(rep)
for r in range(rep):
    gi4_mse[r] = np.mean(p2[:, s1:s][r])
gi4_root_mse = np.zeros(rep)
for r in range(rep):
    gi4_root_mse[r] = pow(gi4_mse[r] - p2_t, 2)
gi4_final = pow(np.sum(gi4_root_mse) / rep, 0.5)

gi5_mse = np.zeros(rep)
for r in range(rep):
    gi5_mse[r] = np.mean(sigma[:, s1:s][r])
gi5_root_mse = np.zeros(rep)
for r in range(rep):
    gi5_root_mse[r] = pow(gi5_mse[r] - sigma_t, 2)
gi5_final = pow(np.sum(gi5_root_mse) / rep, 0.5)

print('beta[0] Mean %f' % np.mean(fi0_mse))
print('beta[0] RMSE %f' % fi0_final)
print('beta[1] Mean %f' % np.mean(fi1_mse))
print('beta[1] RMSE %f' % fi1_final)
print('beta[2] Mean %f' % np.mean(fi2_mse))
print('beta[2] RMSE %f' % fi2_final)
print('beta[3] Mean %f' % np.mean(fi3_mse))
print('beta[3] RMSE %f' % fi3_final)
print('beta[4] Mean %f' % np.mean(fi4_mse))
print('beta[4] RMSE %f' % fi4_final)
print('beta[5] Mean %f' % np.mean(fi5_mse))
print('beta[5] RMSE %f' % fi5_final)
print('beta[6] Mean %f' % np.mean(fi6_mse))
print('beta[6] RMSE %f' % fi6_final)
print('beta[7] Mean %f' % np.mean(fi7_mse))
print('beta[7] RMSE %f' % fi7_final)
print('beta[8] Mean %f' % np.mean(fi8_mse))
print('beta[8] RMSE %f' % fi8_final)

print('p1 Mean %f' % np.mean(gi3_mse))
print('p1 RMSE %f' % gi3_final)
print('p2 Mean %f' % np.mean(gi4_mse))
print('p2 RMSE %f' % gi4_final)
print('sigma Mean %f' % np.mean(gi5_mse))
print('sigma RMSE %f' % gi5_final)
print('WAIC mean %f' % np.mean(wai))
end = time.time()
print('Running time %f' % (end - start))
