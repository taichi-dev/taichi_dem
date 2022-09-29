from pyparsing import White
import taichi as ti
import math
import os

ti.init(arch=ti.gpu)
vec = ti.math.vec3

SAVE_FRAMES = False

window_size = 1024  # Number of pixels of the window
n = 9000  # Number of grains

density = 100.0
stiffness = 8e3
restitution_coef = 0.001
gravity = -9.81
dt = 0.0001  # Larger dt might lead to unstable results.
substeps = 60


@ti.dataclass
class Grain:
    p: vec  # Position
    m: ti.f32  # Mass
    r: ti.f32  # Radius
    v: vec  # Velocity
    a: vec  # Acceleration
    f: vec  # Force


gf = Grain.field(shape=(n, ))

grid_n = 64
grid_size = 1.0 / grid_n  # Simulation domain of size [0, 1]
print(f"Grid size: {grid_n}x{grid_n}")

"""
grain_r_min = 0.002
grain_r_max = 0.003
"""
grain_r = 0.005
assert grain_r * 2 < grid_size

region_height = n / 10
padding = 0.2
region_width = 1.0 - padding * 2


@ti.kernel
def init():
    for i in gf:
        # Spread grains in a restricted area.
        h = i // region_height
        sq = i % region_height
        l = sq * grid_size       
        
        # sorted array
        pos = vec(l // region_width * grid_size, h * grid_size * 2, l % region_width + padding)
        #pos = vec(l // region_width * grid_size, h * grid_size * 2, l % region_width + padding + grid_size * ti.random() * 0.2)

        #  all random 
        #pos = vec(0.5 + ti.random() * 0.5,  ti.random() * 0.5, ti.random() * 0.5)

        gf[i].p = pos
        #gf[i].r = ti.random() * (grain_r_max - grain_r_min) + grain_r_min
        gf[i].r = grain_r
        gf[i].m = density * math.pi * gf[i].r**2


@ti.kernel
def update():
    for i in gf:
        a = gf[i].f / gf[i].m
        gf[i].v += (gf[i].a + a) * dt / 2.0
        gf[i].p += gf[i].v * dt + 0.5 * a * dt**2
        gf[i].a = a

square_width = 1 
@ti.kernel
def apply_bc():
    bounce_coef = 0.3  # Velocity damping
    for i in gf:
        x = gf[i].p[0]
        y = gf[i].p[1]
        z = gf[i].p[2]

        if y - gf[i].r < 0:
            gf[i].p[1] = gf[i].r
            gf[i].v[1] *= -bounce_coef

        elif y + gf[i].r > square_width:
            gf[i].p[1] = square_width - gf[i].r
            gf[i].v[1] *= -bounce_coef

        if x - gf[i].r < 0:
            gf[i].p[0] = gf[i].r
            gf[i].v[0] *= -bounce_coef

        elif x + gf[i].r > square_width:
            gf[i].p[0] = square_width - gf[i].r
            gf[i].v[0] *= -bounce_coef

        if z - gf[i].r < 0:
            gf[i].p[2] = gf[i].r
            gf[i].v[2] *= -bounce_coef

        elif z + gf[i].r > square_width:
            gf[i].p[2] = square_width - gf[i].r
            gf[i].v[2] *= -bounce_coef


@ti.func
def resolve(i, j):
    rel_pos = gf[j].p - gf[i].p
    dist = ti.sqrt(rel_pos[0]**2 + rel_pos[1]**2 + rel_pos[2]**2)
    delta = -dist + gf[i].r + gf[j].r  # delta = d - 2 * r
    if delta > 0:  # in contact
        normal = rel_pos / dist
        f1 = normal * delta * stiffness
        # Damping force
        M = (gf[i].m * gf[j].m) / (gf[i].m + gf[j].m)
        K = stiffness
        C = 2. * (1. / ti.sqrt(1. + (math.pi / ti.log(restitution_coef))**2)
                  ) * ti.sqrt(K * M)
        V = (gf[j].v - gf[i].v) * normal
        f2 = C * V * normal
        gf[i].f += f2 - f1
        gf[j].f -= f2 - f1


list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")

_column_sum_cur = ti.field(dtype=ti.i32, shape=(grid_n, grid_n))

@ti.kernel
def contact(gf: ti.template()):
    '''
    Handle the collision between grains.
    '''
    for i in gf:
        gf[i].f = vec(0., gravity * gf[i].m, 0)  # Apply gravity.
        #if 

    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        grain_count[grid_idx] += 1
    
    # kernel comunicate with global variable ???? this is a bit amazing 
    for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):        
        ti.atomic_add(column_sum[i, j], grain_count[i, j, k])

    # this is because memory mapping can be out of order
    _prefix_sum_cur = 0
    
    for i, j in ti.ndrange(grid_n, grid_n):
       _column_sum_cur[i,j] = prefix_sum[i, j, 0] = ti.atomic_add(_prefix_sum_cur, column_sum[i, j])
    
    #for i, j in ti.ndrange(grid_n, grid_n):
       #_column_sum_cur[i,j] = prefix_sum[i, j, 0] = ti.atomic_add(_prefix_sum_cur, column_sum[i, j])
    #   print(i, j, _column_sum_cur[i,j])
    """
    # case 1 ok
    for i, j in ti.ndrange(grid_n, grid_n):
        #print(i, j ,k)
        for k in range(grid_n):
            ti.atomic_add(_column_sum_cur[i,j], grain_count[i, j, k])                
            linear_idx = i * grid_n * grid_n + j * grid_n + k

            list_head[linear_idx] = _column_sum_cur[i,j]- grain_count[i, j, k]
            list_cur[linear_idx] = list_head[linear_idx]
            list_tail[linear_idx] = _column_sum_cur[i,j]

    """
    

    """
    # case 2 wrong
    for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):
        #print(i, j ,k)        
        ti.atomic_add(_column_sum_cur[i,j], grain_count[i, j, k])    
        linear_idx = i * grid_n * grid_n + j * grid_n + k
        list_head[linear_idx] = _column_sum_cur[i,j]- grain_count[i, j, k]
        list_cur[linear_idx] = list_head[linear_idx]
        list_tail[linear_idx] = _column_sum_cur[i,j]

    """
    
    #"""
    # case 3 test okay
    for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):
        #print(i, j ,k)   
        pre = ti.atomic_add(_column_sum_cur[i,j], grain_count[i, j, k])        
        linear_idx = i * grid_n * grid_n + j * grid_n + k
        list_head[linear_idx] = pre
        list_cur[linear_idx] = list_head[linear_idx]
        list_tail[linear_idx] = _column_sum_cur[i,j]
        #print( i, j, k, pre)
    #"""

    
    # e

    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        linear_idx = grid_idx[0] * grid_n * grid_n + grid_idx[1] * grid_n + grid_idx[2]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i


    # Fast collision detection
    for i in range(n):
        grid_idx = ti.floor(gf[i].p * grid_n, int)
        x_begin = max(grid_idx[0] - 1, 0)
        x_end = min(grid_idx[0] + 2, grid_n)

        y_begin = max(grid_idx[1] - 1, 0)
        y_end = min(grid_idx[1] + 2, grid_n)

        z_begin = max(grid_idx[2] - 1, 0)
        z_end = min(grid_idx[2] + 2, grid_n)
        # todo still serialize
        for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):
            neigh_linear_idx = neigh_i * grid_n * grid_n + neigh_j * grid_n + neigh_k
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):
                j = particle_id[p_idx]
                if i < j:
                    resolve(i, j)


init()

window = ti.ui.Window('DEM', (window_size, window_size), show_window = True, vsync=False)
scene = ti.ui.Scene()
camera = ti.ui.Camera()
camera.position(1.25, 0.5, 1.25)
#camera.position(0.5, 0.5, 0.5)

camera.up(0.0, 0.5, 0.0)
camera.lookat(-0.5, 0.0, 0.0)
camera.fov(70)
scene.set_camera(camera)

canvas = window.get_canvas()

step = 0
movement_speed = 0.001

while window.running:
 
    for s in range(substeps):
        update()
        apply_bc()
        contact(gf)    
    
    camera.track_user_inputs(window, movement_speed=movement_speed, hold_key=ti.ui.LMB)
    scene.set_camera(camera)

    scene.point_light((5.0, 5.0, 5.0), color=(5.0, 5.0, 5.0))

    pos = gf.p

    scene.particles(pos, radius=grain_r)

    canvas.scene(scene)
    window.show()
    step += 1