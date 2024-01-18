# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath
from core.common import PPTypes, PPConfig


@ti.data_oriented
class PPKernels:

    # GPU functions (callable by kernels) ===========================================
    @ti.func
    def custom_mod(self, a, b) -> PPTypes.FLOAT_GPU:
        return a - b * ti.floor(a / b)

    @ti.func
    def angle_to_dir_2D(self, angle) -> PPTypes.VEC2f:
        return timath.normalize(PPTypes.VEC2f(ti.cos(angle), ti.sin(angle)))

    @ti.func
    def world_to_grid_2D(
            self,
            pos_world,
            domain_min,
            domain_max,
            grid_resolution) -> PPTypes.VEC2i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(pos_relative * ti.cast(
            grid_resolution, PPTypes.FLOAT_GPU), PPTypes.INT_GPU)
        return ti.max(PPTypes.VEC2i(0, 0), ti.min(grid_coord, grid_resolution - (1, 1)))

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

    @ti.kernel
    def deposit_relaxation_step_2D(
            self,
            attenuation: PPTypes.FLOAT_GPU,
            current_deposit_index: PPTypes.INT_GPU,
            DEPOSIT_RESOLUTION: PPTypes.VEC2i,
            deposit_field: ti.template()):
        DIFFUSION_WEIGHTS = [1.0, 1.0, 0.707]
        DIFFUSION_WEIGHTS_NORM = (DIFFUSION_WEIGHTS[0] + 4.0 * DIFFUSION_WEIGHTS[1] + 4.0 * DIFFUSION_WEIGHTS[2])
        for cell in ti.grouped(deposit_field):
            # The "beautiful" expression below implements
            # a 3x3 kernel diffusion with manually wrapped addressing
            # Taichi doesn't support modulo for tuples
            # so each dimension is handled separately
            value = DIFFUSION_WEIGHTS[0] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[2] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1])][current_deposit_index]
            deposit_field[cell][1 - current_deposit_index] = (attenuation * value / DIFFUSION_WEIGHTS_NORM)
        return

    @ti.kernel
    def trace_relaxation_step_2D(
            self,
            attenuation: PPTypes.FLOAT_GPU,
            trace_field: ti.template()):
        for cell in ti.grouped(trace_field):
            # Perturb the attenuation by a small factor
            # to avoid accumulating quantization errors
            trace_field[cell][0] *= (attenuation - 0.001 + 0.002 * ti.random(dtype=PPTypes.FLOAT_GPU))
        return

    @ti.kernel
    def agent_step_2D(
                self,
                sense_distance: PPTypes.FLOAT_GPU,
                sense_angle: PPTypes.FLOAT_GPU,
                steering_rate: PPTypes.FLOAT_GPU,
                sampling_exponent: PPTypes.FLOAT_GPU,
                step_size: PPTypes.FLOAT_GPU,
                agent_deposit: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                distance_sampling_distribution: PPTypes.INT_GPU,
                directional_sampling_distribution: PPTypes.INT_GPU,
                directional_mutation_type: PPTypes.INT_GPU,
                deposit_fetching_strategy: PPTypes.INT_GPU,
                agent_boundary_handling: PPTypes.INT_GPU,
                N_DATA: PPTypes.FLOAT_GPU,
                N_AGENTS: PPTypes.FLOAT_GPU,
                DOMAIN_SIZE: PPTypes.VEC2f,
                DOMAIN_MIN: PPTypes.VEC2f,
                DOMAIN_MAX: PPTypes.VEC2f,
                TRACE_RESOLUTION: PPTypes.VEC2i,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                agents_field: ti.template(),
                trace_field: ti.template(),
                deposit_field: ti.template()):
        for agent in ti.ndrange(agents_field.shape[0]):
            pos = PPTypes.VEC2f(0.0, 0.0)
            pos[0], pos[1], angle, weight = agents_field[agent]

            # Generate new mutated direction by perturbing the original
            dir_fwd = self.angle_to_dir_2D(angle)
            angle_mut = angle
            if directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.DISCRETE:
                angle_mut += (1.0 if ti.random(dtype=PPTypes.FLOAT_GPU) > 0.5 else -1.0) * sense_angle
            elif directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.CONE:
                angle_mut += 2.0 * (ti.random(dtype=PPTypes.FLOAT_GPU) - 0.5) * sense_angle
            dir_mut = self.angle_to_dir_2D(angle_mut)

            # Generate sensing distance for the agent, constant or probabilistic
            agent_sensing_distance = sense_distance
            distance_scaling_factor = 1.0
            if distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.EXPONENTIAL:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999)
                # log & pow are unstable in extremes
                distance_scaling_factor = -ti.log(xi)
            elif distance_sampling_distribution == PPConfig.EnumDistanceSamplingDistribution.MAXWELL_BOLTZMANN:
                xi = timath.clamp(ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999)
                # log & pow are unstable in extremes
                distance_scaling_factor = -0.3033 * ti.log((ti.pow(xi + 0.005, -0.4) - 0.9974) / 7.326)
            agent_sensing_distance *= distance_scaling_factor

            # Fetch deposit to guide the agent
            deposit_fwd = 1.0
            deposit_mut = 0.0
            if deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN:
                deposit_fwd = deposit_field[self.world_to_grid_2D(
                    pos + agent_sensing_distance * dir_fwd,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                deposit_mut = deposit_field[self.world_to_grid_2D(
                    pos + agent_sensing_distance * dir_mut,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
            elif deposit_fetching_strategy == PPConfig.EnumDepositFetchingStrategy.NN_PERTURBED:
                # Fetches the deposit by perturbing the original position by small delta
                # Provides cheap stochastic filtering instead of multi-fetch filters
                field_dd = 2.0 * ti.cast(DOMAIN_SIZE[0], PPTypes.FLOAT_GPU) / ti.cast(DEPOSIT_RESOLUTION[0], PPTypes.FLOAT_GPU)
                pos_fwd = pos + agent_sensing_distance * dir_fwd + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_fwd = deposit_field[self.world_to_grid_2D(
                    pos_fwd, PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]
                pos_mut = pos + agent_sensing_distance * dir_mut + (field_dd * ti.random(dtype=PPTypes.FLOAT_GPU) * self.angle_to_dir_2D(2.0 * timath.pi * ti.random(dtype=PPTypes.FLOAT_GPU)))
                deposit_mut = deposit_field[self.world_to_grid_2D(
                    pos_mut,
                    PPTypes.VEC2f(DOMAIN_MIN),
                    PPTypes.VEC2f(DOMAIN_MAX),
                    PPTypes.VEC2i(DEPOSIT_RESOLUTION))][current_deposit_index]

            # Generate new direction for the agent based on the sampled deposit
            angle_new = angle
            if directional_mutation_type == PPConfig.EnumDirectionalMutationType.DETERMINISTIC:
                angle_new = (steering_rate * angle_mut + (1.0-steering_rate) * angle) if (deposit_mut > deposit_fwd) else (angle)
            elif directional_mutation_type == PPConfig.EnumDirectionalMutationType.PROBABILISTIC:
                p_remain = ti.pow(deposit_fwd, sampling_exponent)
                p_mutate = ti.pow(deposit_mut, sampling_exponent)
                mutation_probability = p_mutate / (p_remain + p_mutate)
                angle_new = (steering_rate * angle_mut + (1.0-steering_rate) * angle) if (ti.random(dtype=PPTypes.FLOAT_GPU) < mutation_probability) else (angle)
            dir_new = self.angle_to_dir_2D(angle_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            # Agent behavior at domain boundaries
            if agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(
                    pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0],
                    DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(
                    pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1],
                    DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                        or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                        or pos_new[1] >= DOMAIN_MAX[1]:
                    pos_new[0] = DOMAIN_MIN[0] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[0]
                    pos_new[1] = DOMAIN_MIN[1] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU), 0.001, 0.999) * DOMAIN_SIZE[1]

            agents_field[agent][0] = pos_new[0]
            agents_field[agent][1] = pos_new[1]
            agents_field[agent][2] = angle_new

            # Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_2D(pos_new,
                                                 PPTypes.VEC2f(DOMAIN_MIN),
                                                 PPTypes.VEC2f(DOMAIN_MAX),
                                                 PPTypes.VEC2i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_2D(
                pos_new, PPTypes.VEC2f(DOMAIN_MIN),
                PPTypes.VEC2f(DOMAIN_MAX),
                PPTypes.VEC2i(TRACE_RESOLUTION))
            trace_field[trace_cell][0] += ti.max(
                1.0e-4,
                ti.cast(
                    N_DATA,
                    PPTypes.FLOAT_GPU) / ti.cast(
                        N_AGENTS,
                        PPTypes.FLOAT_GPU)) * weight
        return

    @ti.kernel
    def render_visualization_2D(
                self,
                trace_vis: PPTypes.FLOAT_GPU,
                deposit_vis: PPTypes.FLOAT_GPU,
                current_deposit_index: PPTypes.INT_GPU,
                TRACE_RESOLUTION: PPTypes.VEC2i,
                DEPOSIT_RESOLUTION: PPTypes.VEC2i,
                VIS_RESOLUTION: PPTypes.VEC2i,
                trace_field: ti.template(),
                deposit_field: ti.template(),
                vis_field: ti.template()):
        for x, y in ti.ndrange(vis_field.shape[0], vis_field.shape[1]):
            deposit_val = deposit_field[
                x * DEPOSIT_RESOLUTION[0] // VIS_RESOLUTION[0],
                y * DEPOSIT_RESOLUTION[1] // VIS_RESOLUTION[1]][current_deposit_index]
            trace_val = trace_field[
                x * TRACE_RESOLUTION[0] // VIS_RESOLUTION[0],
                y * TRACE_RESOLUTION[1] // VIS_RESOLUTION[1]]
            vis_field[x, y] = ti.pow(
                PPTypes.VEC3f(
                    trace_vis * trace_val,
                    deposit_vis * deposit_val,
                    ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)), 1.0/2.2)
        return
