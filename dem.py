import taichi as ti
import math

ti.init(arch=ti.gpu)
vec = ti.math.vec2

bsize = 1280  # Window size
n = 4096  # Number of grains
density = 1000.0
stiffness = 4e7
restitution_coef = 0.1
gravity = -9.81
dt = 0.0002  # Larger dt might lead to unstable results.
substeps = 50


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force


gf = Grain.field(shape=(n, ))


@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * 18
        padding = 100
        block = bsize - padding
        offset = vec(l % block + padding / 2, l // block * 20 + padding)
        offset /= bsize
        gf[i].p = vec(0, 0) + offset
        gf[i].r = ti.random() * 2 + 1
        gf[i].m = density * 2 * math.pi * gf[i].r


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a


@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0] * bsize
        y = gf[i].p[1] * bsize

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r / bsize
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > bsize:
            gf[i].p[1] = (bsize - gf[i].r) / bsize
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r / bsize
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > bsize:
            gf[i].p[0] = (bsize - gf[i].r) / bsize
            gf[i].v[0] *= -bounce_coef


@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    # Brute-force traversing
    for i in range(n):
        for j in range(i + 1, n):
            rel_pos = (gf[j].p - gf[i].p) * bsize
            dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
            if delta > 0:  # in contact
                normal = rel_pos / dist
                f1 = normal * delta * stiffness
                gf[i].f -= f1
                gf[j].f += f1
                # Damping force
                M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
                K = stiffness
                C = 2. * (1. / ti.sqrt(1. +
                                       (math.pi / ti.log(restitution_coef))**2)
                          ) * ti.sqrt(K * M)
                V = (gf[j].v - gf[i].v) * normal
                f2 = C * V * normal
                gf[i].f += f2
                gf[j].f -= f2


init()
gui = ti.GUI('Taichi DEM', (bsize, bsize))
while gui.running:
    for s in range(substeps):
        update()
        apply_bc()
        contact(gf)
    pos = gf.p.to_numpy()
    r = gf.r.to_numpy()
    gui.circles(pos, radius=r)
    gui.show()
