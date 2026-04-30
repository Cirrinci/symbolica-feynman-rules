"""Shared direct-API symbols and operator dictionaries for tests."""

from symbolica import S

from symbolic.spenso_structures import SPINOR_KIND, gamma_matrix


x = S("x")
d = S("d")

p1, p2, p3, p4 = S("p1", "p2", "p3", "p4")
b1, b2, b3, b4 = S("b1", "b2", "b3", "b4")

phi0 = S("phi0")
phiC0 = S("phiC0")
phiCdag0 = S("phiCdag0")
psibar0, psi0 = S("psibar0", "psi0")
A0 = S("A0")

lamC = S("lamC")
yF = S("yF")
g4F = S("g4F")

alpha_s = S("alpha_s")
i_psi_bar, i_psi = S("i_psi_bar", "i_psi")
mu = S("mu")
s1, s2, s3 = S("s1", "s2", "s3")

L_yukawa = dict(
    coupling=yF,
    alphas=[psibar0, psi0, phi0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[alpha_s, alpha_s, None],
    leg_spins=[s1, s2, s3],
)

L_vec_current = dict(
    coupling=gamma_matrix(i_psi_bar, i_psi, mu),
    alphas=[psibar0, psi0, A0],
    betas=[b1, b2, b3],
    ps=[p1, p2, p3],
    statistics="fermion",
    field_roles=["psibar", "psi", "scalar"],
    leg_roles=["psibar", "psi", "scalar"],
    field_spinor_indices=[i_psi_bar, i_psi, None],
    leg_spins=[s1, s2, s3],
)

L_4fermion = dict(
    coupling=g4F,
    alphas=[psi0, psibar0, psi0, psibar0],
    betas=[b1, b2, b3, b4],
    ps=[p1, p2, p3, p4],
    statistics="fermion",
    field_roles=["psi", "psibar", "psi", "psibar"],
    leg_roles=["psi", "psibar", "psi", "psibar"],
)
