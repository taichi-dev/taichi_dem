import taichi as ti
import math

ti.init(arch=ti.gpu, debug=True)
vec = ti.math.vec2

bsize = 1200  # Window size
n = 4096 * 4  # Number of grains
# n = 4096
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

grid_size = 6
assert bsize % grid_size == 0
grid_n = bsize // grid_size
print(f"Grid size: {grid_n}x{grid_n}")


@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        l = i * grid_size
        padding = 100
        block = bsize - padding
        offset = vec(l % block + padding / 2, l // block * grid_size + padding)
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


@ti.func
def resolve(i, j):
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
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2
        gf[j].f -= f2


index_pos = ti.field(dtype=ti.i32, shape=grid_n * grid_n + 1)
index_current_pos = ti.field(dtype=ti.i32, shape=grid_n * grid_n + 1)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=grid_n, name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")


@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m)  # Apply gravity.

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * (bsize / grid_size), int)
        grain_count[grid_idx] += 1

    for i in range(grid_n):
        sum = 0
        for j in range(grid_n):
            sum += grain_count[i, j]
        column_sum[i] = sum

    ti.loop_config(serialize=True)
    for i in range(1, grid_n):
        prefix_sum[i, 0] = prefix_sum[i - 1, 0] + column_sum[i - 1]

    for i in range(grid_n):
        for j in range(grid_n):
            if j > 0:
                prefix_sum[i, j] = prefix_sum[i, j - 1] + grain_count[i, j]
            index_pos[i * grid_n + j + 1] = prefix_sum[i, j]
            index_current_pos[i * grid_n + j + 1] = prefix_sum[i, j]

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * (bsize / grid_size), int)
        linear_idx = grid_idx[0] * grid_n + grid_idx[1]
        grain_location = ti.atomic_add(index_current_pos[linear_idx], 1)
        particle_id[grain_location] = i

    # Brute-force traversing
    '''
    for i in range(n):
        for j in range(i + 1, n):
            resolve(i, j)
    '''

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * (bsize / grid_size), int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        for neigh_i in range(x_begin, x_end):
            for neigh_j in range(y_begin, y_end):
                neigh_linear_idx = neigh_i * grid_n + neigh_j
                for p_idx in range(index_pos[neigh_linear_idx],
                                   index_pos[neigh_linear_idx + 1]):
                    j = particle_id[p_idx]
                    if i < j:
                        resolve(i, j)


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

# TODO: angular momentum
# TODO: use simulation domain [0, 1] instead of [0, bsize]
