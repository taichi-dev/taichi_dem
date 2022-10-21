import taichi as ti
import math
import os

_fp = ti.f32
ti.init(arch=ti.gpu) 

SAVE_FRAMES = False

vec3f = ti.types.vector(3, _fp)
gravity = vec3f(0, -9.8, 0)
dt = 0.01
numSteps = 5
sdt = dt / numSteps
n = 10000

maxX = 1
maxZ = maxX
maxY = maxX
particleRadius = 0.01
maxVel = 0.5 * particleRadius
kernelRadius = 3.0 * particleRadius
particleDiameter = 2 * particleRadius
restDensity = 1 / (particleDiameter * particleDiameter)

viscosity = 0.01
h = kernelRadius
h2 = h * h
PI = 3.14
kernelScale = 4.0 / (PI * h2 * h2 * h2 * h2);		

pos = ti.Vector.field(3, dtype=_fp, shape=n)
prepos = ti.Vector.field(3, dtype=_fp, shape=n)
vel = ti.Vector.field(3, dtype=_fp, shape=n)

eps = 1e-5
grid_size = 0.05
grid_n = math.ceil(1 / grid_size)

print(f"Grid size: {grid_n}x{grid_n}")

assert particleRadius * 2 < grid_size

list_head = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_cur = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)
list_tail = ti.field(dtype=ti.i32, shape=grid_n * grid_n * grid_n)

grain_count = ti.field(dtype=ti.i32,
                       shape=(grid_n, grid_n, grid_n),
                       name="grain_count")
column_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="column_sum")
prefix_sum = ti.field(dtype=ti.i32, shape=(grid_n, grid_n), name="prefix_sum")
particle_id = ti.field(dtype=ti.i32, shape=n, name="particle_id")

@ti.func 
def tougong(i:int, sdt: float, step: int):
    vel[i] += gravity * sdt
    #"""
    _toCenter = pos[i] - vec3f(0.5, 0.5, 0.5)          
    _norm = _toCenter.norm()          
    if step > 500:
        if _norm < 0.1:
            vel[i] += _toCenter.normalized() * (1 - _norm) * 1000
            
    if 400 > step > 100: 
        _rotateforce  = vec3f(_toCenter[2], 0, -_toCenter[0]).normalized() 
        vel[i] += -_toCenter.normalized() * (1 - _norm)  * 50 * sdt
        vel[i] += _rotateforce * (0.5 - abs(0.5 - pos[i][1])) * 5 * sdt
    #"""
    return vel[i]

@ti.func
def getNeighborGrid(i):
    grid_idx = ti.floor(pos[i]/maxX * grid_n, int)
    x_begin = max(grid_idx[0] - 1, 0)
    x_end = min(grid_idx[0] + 2, grid_n)
    y_begin = max(grid_idx[1] - 1, 0)
    y_end = min(grid_idx[1] + 2, grid_n) 
    z_begin = max(grid_idx[2] - 1, 0)
    z_end = min(grid_idx[2] + 2, grid_n)
    return x_begin, x_end, y_begin, y_end, z_begin, z_end 

@ti.func
def getDensityAndNormal(_norm: float, dist: ti.template()):
    r2 = _norm * _norm 
    w = (h2 - r2) 
    if _norm > 0:
        dist = dist.normalized()
    return w, dist

@ti.func
def calculateGrad(w: float, _norm: float):    
    return  (kernelScale * 3 * w * w * (-2.0 * _norm)) / restDensity

@ti.func
def findNeighbors():
    grain_count.fill(0)

    for i in range(n):
        grid_idx = ti.floor(pos[i]* grid_n, int)
        grain_count[grid_idx] += 1
    
    column_sum.fill(0)
    # kernel comunicate with global variable ???? this is a bit amazing 
    for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):        
        ti.atomic_add(column_sum[i, j], grain_count[i, j, k])

    # this is because memory mapping can be out of order
    _prefix_sum_cur = 0    
    for i, j in ti.ndrange(grid_n, grid_n):
        prefix_sum[i, j] = ti.atomic_add(_prefix_sum_cur, column_sum[i, j])
    
        
    for i, j, k in ti.ndrange(grid_n, grid_n, grid_n):        
        # we cannot visit prefix_sum[i,j] in this loop
        pre = ti.atomic_add(prefix_sum[i,j], grain_count[i, j, k])        
        linear_idx = i * grid_n * grid_n + j * grid_n + k
        list_head[linear_idx] = pre
        list_cur[linear_idx] = list_head[linear_idx]
        # only pre pointer is useable
        list_tail[linear_idx] = pre + grain_count[i, j, k]       

    for i in range(n):
        grid_idx = ti.floor(pos[i] * grid_n, int)
        linear_idx = grid_idx[0] * grid_n * grid_n + grid_idx[1] * grid_n + grid_idx[2]
        grain_location = ti.atomic_add(list_cur[linear_idx], 1)
        particle_id[grain_location] = i

@ti.func
def solveBoundaries():
    # if there is not random it will be very unstable at border
    for i in range(n):
        if pos[i][1] <= 0:
            pos[i][1] = eps * ti.random()
        if pos[i][1] >= maxY:
            pos[i][1] = maxY - eps * ti.random(); 
        if (pos[i][0] <= 0): 
            pos[i][0] = eps * ti.random()
        if (pos[i][0] >= maxX):
            pos[i][0] = maxX - eps * ti.random(); 
        if (pos[i][2] <= 0): 
            pos[i][2] = eps * ti.random()
        if (pos[i][2] >= maxZ):
            pos[i][2] = maxZ - eps * ti.random(); 

@ti.func
def applyViscosity(i, sdt):
    avgVel = vec3f(0, 0, 0)
    _count = 0    
    x_begin, x_end, y_begin, y_end, z_begin, z_end = getNeighborGrid(i)
    for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):            
            neigh_linear_idx = neigh_i * grid_n * grid_n + neigh_j * grid_n + neigh_k
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):                    
                j = particle_id[p_idx]
                _dist = pos[i] - pos[j]
                if _dist.norm() < h:			
                    avgVel += vel[j]
                    _count += 1
    # averge nearby particles
    if _count > 0:                
        avgVel /= _count        
        _delta = avgVel -  vel[i]        
        vel[i] += viscosity * _delta

@ti.func
def solveFluid():

    #avgRho = 0.0 for debug
    for i in range(n):
        rho = 0.0
        sumGrad2 = 0.0        
        _gradient = vec3f(0, 0, 0)

        x_begin, x_end, y_begin, y_end, z_begin, z_end = getNeighborGrid(i)      
        for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):            
            neigh_linear_idx = neigh_i * grid_n * grid_n + neigh_j * grid_n + neigh_k
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):
                j = particle_id[p_idx]                
                _dist = pos[j] - pos[i]
                _norm = _dist.norm()
                if _norm < h:
                    w, _dist = getDensityAndNormal(_norm, _dist)
                    _grad = calculateGrad(w, _norm)                
                    _gradient -= _grad * _dist 
                    sumGrad2 += _grad * _grad
                    rho += kernelScale * w * w * w
        
        sumGrad2 += _gradient.dot(_gradient)
        #avgRho += rho
        
        C = rho / restDensity - 1.0
        if C < 0.0:
            continue
        _lambda = -C / (sumGrad2 + 0.0001)

        for neigh_i, neigh_j, neigh_k in ti.ndrange((x_begin,x_end),(y_begin,y_end),(z_begin,z_end)):
            neigh_linear_idx = neigh_i * grid_n * grid_n + neigh_j * grid_n + neigh_k
            for p_idx in range(list_head[neigh_linear_idx],
                            list_tail[neigh_linear_idx]):
                j = particle_id[p_idx]		
                if (j == i) :
                    pos[j] += _lambda * _gradient
                else:
                    _dist = pos[j] - pos[i]
                    _norm = _dist.norm(eps=0)
                    _grad = 0.0
                    if _norm < h:
                        w, _dist = getDensityAndNormal(_norm, _dist)   
                        _grad = calculateGrad(w, _norm)                
                    pos[j] += _lambda * _grad * _dist

@ti.kernel
def init():
    _w = 20 
    _h = 20
    for i in range(n):
        _y = i // (_h * _w)
        _cur = i % (_h * _w)
        #pos[i] = 0.03 * vec3f(_cur%_w, _y, _cur//_w) + vec3f(0.1, 0.1, 0.1)
        pos[i] = 0.03 * vec3f(_cur%_w + ti.random(), _y, _cur//_w + ti.random()) + vec3f(0.1, 0.1, 0.1)


@ti.kernel
def update(step: int):

    # predict 
    for i in range(n):
        vel[i] = tougong(i, sdt, step)
        prepos[i] = pos[i]
        pos[i] += vel[i] * sdt

    # solve
    solveBoundaries()

    findNeighbors()    
    solveFluid()

    # derive velocities
    for i in range(n):
        deltaV = pos[i] - prepos[i]

        # CFL
        _Vnorm = deltaV.norm()
        if _Vnorm > maxVel:
            deltaV *= maxVel / _Vnorm
            pos[i] = prepos[i] + deltaV
    
        vel[i] = deltaV / sdt
        
        applyViscosity(i, sdt)

win_x = 640
win_y = 640

window = ti.ui.Window("pbf 3d", 
(win_x, win_y), vsync=True
)
canvas = window.get_canvas()
canvas.set_background_color((0, 0, 0))
scene = ti.ui.Scene()

camera = ti.ui.make_camera()
camera.position(2.5, 1, 2)
camera.lookat(0, 0, 0)
scene.ambient_light((0.5, 0.5, 0.5))
scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))

step = 0
init()
while window.running:
    
    camera.track_user_inputs(window, movement_speed=0.05, hold_key=ti.ui.RMB)
 
    scene.set_camera(camera)
    
    for s in range(numSteps):
        update(step)    
    
    scene.particles(pos, color = (0, 1, 1), radius = particleRadius)

    canvas.scene(scene)
    
    if step % 5 == 0 and SAVE_FRAMES:
        window.save_image(f"outputs/{step:06}.png")

    if step > 600:
        break
    
    window.show()

    step +=1 
