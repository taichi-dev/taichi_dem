import taichi as ti

ti.init(arch=ti.cpu)
vec = ti.math.vec2
bsize = 640 # Window size

@ti.dataclass
class Grain:
    p: vec     # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec     # Velocity
    a: vec     # Acceleration
    f: vec     # Force
    
gf = Grain.field(shape=(100,))

@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        offset = ti.Vector([ti.random(), ti.random()]) * 0.8 + 0.1
        gf[i].p = vec(0,0) + offset
        gf[i].r = ti.random() * 20 + 5
        gf[i].m = 1.0

@ti.kernel
def apply_gravity():
    for i in gf:
        gf[i].f += vec(0., -9.81*gf[i].m)

@ti.kernel
def update():
    dt = 0.005
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a

@ti.kernel
def apply_bc():
    for i in gf:
        y = gf[i].p[1] * bsize
        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r / bsize
            gf[i].v[1] *= - 0.9

@ti.kernel
def contact():
    '''
    Handle the contact of grains.
    '''
    pass

init()
apply_gravity()
gui = ti.GUI('DEM', (bsize, bsize))
while gui.running:
    update()
    apply_bc()
    pos = gf.p.to_numpy()
    r   = gf.r.to_numpy()
    gui.circles(pos, radius=r)
    gui.show()
