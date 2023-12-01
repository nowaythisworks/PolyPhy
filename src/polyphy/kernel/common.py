# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath

from core.common import PPTypes


@ti.data_oriented
class PPKernels:

    # GPU functions (callable by kernels) ===========================================
    @ti.func
    def custom_mod(self, a, b) -> PPTypes.FLOAT_GPU:
        return a - b * ti.floor(a / b)

    @ti.func
    def ray_AABB_intersection(self, ray_pos, ray_dir, AABB_min, AABB_max):
        t0 = (AABB_min[0] - ray_pos[0]) / ray_dir[0]
        t1 = (AABB_max[0] - ray_pos[0]) / ray_dir[0]
        t2 = (AABB_min[1] - ray_pos[1]) / ray_dir[1]
        t3 = (AABB_max[1] - ray_pos[1]) / ray_dir[1]
        t4 = (AABB_min[2] - ray_pos[2]) / ray_dir[2]
        t5 = (AABB_max[2] - ray_pos[2]) / ray_dir[2]
        t6 = ti.max(ti.max(ti.min(t0, t1), ti.min(t2, t3)), ti.min(t4, t5))
        t7 = ti.min(ti.min(ti.max(t0, t1), ti.max(t2, t3)), ti.max(t4, t5))
        return PPTypes.VEC2f(-1.0, -1.0) if (t7 < 0.0 or t6 >= t7) else PPTypes.VEC2f(t6, t7)

    # GPU functions (path-tracer specific) ==========================================
    @ti.func
    def trace_to_rho(self, trace):
        sample_weight = 0.01
        trim_density = 1.0e-5
        ambient_trace = 0.0
        return sample_weight * (ti.max(trace - trim_density, 0.0) + ambient_trace)

    @ti.func
    def sample_volume(self, pos, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis, deposit_vis):
        deposit_val = deposit_field[self.world_to_grid_3D(pos, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(
            DOMAIN_MAX), PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
        trace_val = trace_field[self.world_to_grid_3D(pos, PPTypes.VEC3f(
            DOMAIN_MIN), PPTypes.VEC3f(DOMAIN_MAX), PPTypes.VEC3i(TRACE_RESOLUTION))][0]
        density = (trace_val * trace_vis + deposit_val * deposit_vis) * sigma_e / 10

        # return VEC3f(deposit_val, deposit_val, deposit_val)
        return PPTypes.VEC3f(trace_vis * trace_val, deposit_vis * deposit_val, density)
        # ^^  the blue is density, covering most of the screen. reduce it.

    @ti.func
    def get_rho(self, ray_pos, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis=1, deposit_vis=0):
        return (self.sample_volume(ray_pos, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis, deposit_vis)[0])


    @ti.func
    def delta_step(self, sigma_max_inv, xi):
        return -ti.log(ti.max(xi, 0.001)) * sigma_max_inv


    @ti.func
    def delta_tracking(self, ray_pos, ray_dir, t_min, t_max, rho_max_inv, sigma_a, sigma_s, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis=1, deposit_vis=0):
        sigma_max_inv = rho_max_inv / (sigma_a + sigma_s)
        t = t_min
        event_rho = 0.0

        while (t <= t_max and ti.random(dtype=PPTypes.FLOAT_GPU) > event_rho * rho_max_inv):
            t += self.delta_step(sigma_max_inv, ti.random(dtype=PPTypes.FLOAT_GPU))
            event_rho = self.get_rho((ray_pos + t * ray_dir), DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis, deposit_vis)

        return t

    # this function exists in case we want to implement a middleman step between colormap and emission, such as a non-uniform scale for emission values
    @ti.func
    def get_emitted_trace_L(self, pos: PPTypes.FLOAT_GPU, colormap: ti.template()):
        return self.TexSamplePosition(pos, colormap)

    # this function maps any point value to a color on a 1x123 colormap
    @ti.func
    def TexSamplePosition(self, pos: PPTypes.FLOAT_GPU, colormap: ti.template(), shift: PPTypes.INT_GPU = 0):
        # find an index
        xPos = timath.clamp(ti.min(ti.floor((pos) * 123, ti.i32), 122 - 1) + shift, 0, 122 - 1)
        red = colormap[xPos, 1, 0]
        green = colormap[xPos, 1, 1]
        blue = colormap[xPos, 1, 2]
        return [red, green, blue]

    @ti.func
    def get_dir_1(self, dir):
        inv_norm = 1.0 / ti.sqrt(dir[0] * dir[0] + dir[2] * dir[2])
        return PPTypes.VEC3f(dir[2] * inv_norm, 0.0, -dir[0] * inv_norm)

    @ti.func
    def get_dir_2(self, dir, v1):
        return ti.math.cross(dir, v1)

    @ti.func
    def sample_HG(self, v, g):
        xi = ti.random()
        cos_theta = 0.0
        if (ti.abs(g) > 1.e-3):
            sqr_term = (1.0 - g * g) / (1.0 - g + 2.0 * g * xi)
            cos_theta = (1.0 + g * g - sqr_term * sqr_term) / (2.0 * abs(g))
        else:
            cos_theta = 1.0 - 2.0 * xi

        sin_theta = ti.sqrt(ti.max(0.0, 1.0 - cos_theta * cos_theta))
        phi = (ti.math.pi * 2) * ti.random()

        v1 = self.get_dir_1(v)
        v2 = self.get_dir_2(v, v1)

        return sin_theta * ti.cos(phi) * v1 + sin_theta * ti.sin(phi) * v2 + cos_theta * v

    # GPU kernels (callable by core classes via Taichi API) ========================
    @ti.kernel
    def zero_field(self, f: ti.template()):
        for cell in ti.grouped(f):
            f[cell].fill(0.0)
        return

    @ti.kernel
    def copy_field(self, dst: ti.template(), src: ti.template()):
        for cell in ti.grouped(dst):
            dst[cell] = src[cell]
        return
