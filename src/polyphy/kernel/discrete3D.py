# PolyPhy
# License: https://github.com/PolyPhyHub/PolyPhy/blob/main/LICENSE
# Author: Oskar Elek
# Maintainers:

import taichi as ti
import taichi.math as timath

from core.common import PPTypes, PPConfig
from .common import PPKernels


@ti.data_oriented
class PPKernels_3DDiscrete(PPKernels):
    @ti.func
    def world_to_grid_3D(
        self,
        pos_world,
        domain_min,
        domain_max,
            grid_resolution) -> PPTypes.VEC3i:
        pos_relative = (pos_world - domain_min) / (domain_max - domain_min)
        grid_coord = ti.cast(
            pos_relative * ti.cast(
                grid_resolution,
                PPTypes.FLOAT_GPU), PPTypes.INT_GPU)
        return ti.max(PPTypes.VEC3i(0, 0, 0), ti.min(grid_coord, grid_resolution - (1, 1, 1)))

    @ti.func
    def angles_to_dir_3D(self, theta, phi) -> PPTypes.VEC3f:
        return timath.normalize(PPTypes.VEC3f(ti.sin(theta) * ti.cos(phi),
                                              ti.cos(theta),
                                              ti.sin(theta) * ti.sin(phi)))

    @ti.func
    def dir_3D_to_angles(self, dir):
        theta = timath.acos(dir[1] / timath.length(dir))
        phi = timath.atan2(dir[2], dir[0])
        return theta, phi

    @ti.func
    def axial_rotate_3D(self, vec, axis, angle):
        return (ti.cos(angle) * vec + ti.sin(angle) * (timath.cross(axis, vec)) +
                timath.dot(axis, vec) * (1.0 - ti.cos(angle)) * axis)

    @ti.kernel
    def data_step_3D_discrete(
        self,
        data_deposit: PPTypes.FLOAT_GPU,
        current_deposit_index: PPTypes.INT_GPU,
        DOMAIN_MIN: PPTypes.VEC3f,
        DOMAIN_MAX: PPTypes.VEC3f,
        DEPOSIT_RESOLUTION: PPTypes.VEC3i,
        data_field: ti.template(),
            deposit_field: ti.template()):
        for point in ti.ndrange(data_field.shape[0]):
            pos = PPTypes.VEC3f(0.0, 0.0, 0.0)
            pos[0], pos[1], pos[2], weight = data_field[point]
            deposit_cell = self.world_to_grid_3D(
                pos,
                PPTypes.VEC3f(DOMAIN_MIN),
                PPTypes.VEC3f(DOMAIN_MAX),
                PPTypes.VEC3i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += data_deposit * weight
        return

    @ti.kernel
    def agent_step_3D_discrete(
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
        DOMAIN_SIZE: PPTypes.VEC3f,
        DOMAIN_MIN: PPTypes.VEC3f,
        DOMAIN_MAX: PPTypes.VEC3f,
        TRACE_RESOLUTION: PPTypes.VEC3i,
        DEPOSIT_RESOLUTION: PPTypes.VEC3i,
        agents_field: ti.template(),
        trace_field: ti.template(),
            deposit_field: ti.template()):
        for agent in ti.ndrange(agents_field.shape[0]):
            pos = PPTypes.VEC3f(0.0, 0.0, 0.0)
            pos[0], pos[1], pos[2], theta, phi, weight = agents_field[agent]

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

            # Generate new mutated direction by perturbing the original
            # TODO implement the other sampling strategies
            dir_fwd = self.angles_to_dir_3D(theta, phi)
            xi_dir = 1.0
            if directional_sampling_distribution == PPConfig.EnumDirectionalSamplingDistribution.CONE:
                xi_dir = ti.random(dtype=PPTypes.FLOAT_GPU)
            theta_sense = theta - xi_dir * sense_angle
            off_fwd_dir = self.angles_to_dir_3D(theta_sense, phi)
            random_azimuth = ti.random(dtype=PPTypes.FLOAT_GPU) * 2.0 * timath.pi - timath.pi
            dir_mut = self.axial_rotate_3D(off_fwd_dir, dir_fwd, random_azimuth)

            # Fetch deposit to guide the agent
            # TODO implement the other mutation strategies
            deposit_fwd = deposit_field[self.world_to_grid_3D(
                pos + agent_sensing_distance * dir_fwd,
                PPTypes.VEC3f(DOMAIN_MIN),
                PPTypes.VEC3f(DOMAIN_MAX),
                PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
            deposit_mut = deposit_field[self.world_to_grid_3D(
                pos + agent_sensing_distance * dir_mut,
                PPTypes.VEC3f(DOMAIN_MIN),
                PPTypes.VEC3f(DOMAIN_MAX),
                PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]

            # Generate new direction for the agent based on the sampled deposit
            p_remain = ti.pow(deposit_fwd, sampling_exponent)
            p_mutate = ti.pow(deposit_mut, sampling_exponent)
            mutation_probability = p_mutate / (p_remain + p_mutate)
            dir_new = dir_fwd
            theta_new = theta
            phi_new = phi
            if p_remain + p_mutate > 1.0e-5:
                if ti.random(dtype=PPTypes.FLOAT_GPU) < mutation_probability:
                    theta_mut = theta - steering_rate * xi_dir * sense_angle
                    off_mut_dir = self.angles_to_dir_3D(theta_mut, phi)
                    dir_new = self.axial_rotate_3D(off_mut_dir, dir_fwd, random_azimuth)
                    theta_new, phi_new = self.dir_3D_to_angles(dir_new)
            pos_new = pos + step_size * distance_scaling_factor * dir_new

            # Agent behavior at domain boundaries
            if agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.WRAP:
                pos_new[0] = self.custom_mod(
                    pos_new[0] - DOMAIN_MIN[0] + DOMAIN_SIZE[0],
                    DOMAIN_SIZE[0]) + DOMAIN_MIN[0]
                pos_new[1] = self.custom_mod(
                    pos_new[1] - DOMAIN_MIN[1] + DOMAIN_SIZE[1],
                    DOMAIN_SIZE[1]) + DOMAIN_MIN[1]
                pos_new[2] = self.custom_mod(
                    pos_new[2] - DOMAIN_MIN[2] + DOMAIN_SIZE[2],
                    DOMAIN_SIZE[2]) + DOMAIN_MIN[2]
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_CENTER:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                    or pos_new[1] >= DOMAIN_MAX[1] \
                    or pos_new[2] <= DOMAIN_MIN[2] \
                        or pos_new[2] >= DOMAIN_MAX[2]:
                    pos_new[0] = 0.5 * (DOMAIN_MIN[0] + DOMAIN_MAX[0])
                    pos_new[1] = 0.5 * (DOMAIN_MIN[1] + DOMAIN_MAX[1])
                    pos_new[2] = 0.5 * (DOMAIN_MIN[2] + DOMAIN_MAX[2])
            elif agent_boundary_handling == PPConfig.EnumAgentBoundaryHandling.REINIT_RANDOMLY:
                if pos_new[0] <= DOMAIN_MIN[0] \
                    or pos_new[0] >= DOMAIN_MAX[0] \
                    or pos_new[1] <= DOMAIN_MIN[1] \
                    or pos_new[1] >= DOMAIN_MAX[1] \
                    or pos_new[2] <= DOMAIN_MIN[2] \
                        or pos_new[2] >= DOMAIN_MAX[2]:
                    pos_new[0] = DOMAIN_MIN[0] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU),
                        0.001, 0.999) * DOMAIN_SIZE[0]
                    pos_new[1] = DOMAIN_MIN[1] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU),
                        0.001, 0.999) * DOMAIN_SIZE[1]
                    pos_new[2] = DOMAIN_MIN[2] + timath.clamp(
                        ti.random(dtype=PPTypes.FLOAT_GPU),
                        0.001, 0.999) * DOMAIN_SIZE[2]

            agents_field[agent][0] = pos_new[0]
            agents_field[agent][1] = pos_new[1]
            agents_field[agent][2] = pos_new[2]
            agents_field[agent][3] = theta_new
            agents_field[agent][4] = phi_new

            # Generate deposit and trace at the new position
            deposit_cell = self.world_to_grid_3D(
                pos_new,
                PPTypes.VEC3f(DOMAIN_MIN),
                PPTypes.VEC3f(DOMAIN_MAX),
                PPTypes.VEC3i(DEPOSIT_RESOLUTION))
            deposit_field[deposit_cell][current_deposit_index] += agent_deposit * weight

            trace_cell = self.world_to_grid_3D(
                pos_new,
                PPTypes.VEC3f(DOMAIN_MIN),
                PPTypes.VEC3f(DOMAIN_MAX),
                PPTypes.VEC3i(TRACE_RESOLUTION))
            trace_field[trace_cell][0] += weight
        return

    @ti.kernel
    def deposit_relaxation_step_3D_discrete(
        self,
        attenuation: PPTypes.FLOAT_GPU,
        current_deposit_index: PPTypes.INT_GPU,
        DEPOSIT_RESOLUTION: PPTypes.VEC3i,
            deposit_field: ti.template()):
        DIFFUSION_WEIGHTS = [1.0, 1.0, 0.0, 0.0]
        DIFFUSION_WEIGHTS_NORM = (DIFFUSION_WEIGHTS[0] + 6.0 * DIFFUSION_WEIGHTS[1] + 12.0 * DIFFUSION_WEIGHTS[2] + 8.0 * DIFFUSION_WEIGHTS[3])
        for cell in ti.grouped(deposit_field):
            # The "beautiful" expression below implements
            # a 3x3x3 kernel diffusion in a 6-neighborhood
            # with manually wrapped addressing
            # Taichi doesn't support modulo for tuples
            # so each dimension is handled separately
            value = DIFFUSION_WEIGHTS[0] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] - 1 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] - 1 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 0 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] + 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]\
                  + DIFFUSION_WEIGHTS[1] * deposit_field[((cell[0] + 0 + DEPOSIT_RESOLUTION[0]) % DEPOSIT_RESOLUTION[0], (cell[1] + 0 + DEPOSIT_RESOLUTION[1]) % DEPOSIT_RESOLUTION[1], (cell[2] - 1 + DEPOSIT_RESOLUTION[2]) % DEPOSIT_RESOLUTION[2])][current_deposit_index]
            deposit_field[cell][1 - current_deposit_index] = (attenuation * value / DIFFUSION_WEIGHTS_NORM)
        return

    @ti.kernel
    def trace_relaxation_step_3D_discrete(
        self,
        attenuation: PPTypes.FLOAT_GPU,
            trace_field: ti.template()):
        for cell in ti.grouped(trace_field):
            # Perturb the attenuation by a small factor
            # to avoid accumulating quantization errors
            trace_field[cell][0] *= attenuation - 0.001 + 0.002 * ti.random(
                dtype=PPTypes.FLOAT_GPU)
        return

    @ti.kernel
    def render_visualization_3D_raymarched(
        self,
        trace_vis: PPTypes.FLOAT_GPU,
        deposit_vis: PPTypes.FLOAT_GPU,
        camera_distance: PPTypes.FLOAT_GPU,
        camera_polar: PPTypes.FLOAT_GPU,
        camera_azimuth: PPTypes.FLOAT_GPU,
        n_ray_steps_f: PPTypes.FLOAT_GPU,
        current_deposit_index: PPTypes.INT_GPU,
        TRACE_RESOLUTION: PPTypes.VEC3i,
        DEPOSIT_RESOLUTION: PPTypes.VEC3i,
        VIS_RESOLUTION: PPTypes.VEC2i,
        DOMAIN_SIZE_MAX: PPTypes.FLOAT_GPU,
        DOMAIN_MIN: PPTypes.VEC3f,
        DOMAIN_MAX: PPTypes.VEC3f,
        DOMAIN_CENTER: PPTypes.VEC3f,
        RAY_EPSILON: PPTypes.FLOAT_GPU,
        deposit_field: ti.template(),
        trace_field: ti.template(),
        vis_field: ti.template(),
        colormap: ti.template()):

            n_steps = ti.cast(n_ray_steps_f, PPTypes.INT_GPU)
            aspect_ratio = ti.cast(
                VIS_RESOLUTION[0],
                PPTypes.FLOAT_GPU) / ti.cast(
                    VIS_RESOLUTION[1],
                    PPTypes.FLOAT_GPU)
            screen_distance = DOMAIN_SIZE_MAX
            camera_offset = camera_distance * PPTypes.VEC3f(
                ti.cos(camera_azimuth) * ti.sin(camera_polar),
                ti.sin(camera_azimuth) * ti.sin(camera_polar),
                ti.cos(camera_polar))
            camera_pos = DOMAIN_CENTER + camera_offset
            cam_Z = timath.normalize(-camera_offset)
            cam_Y = PPTypes.VEC3f(0.0, 0.0, 1.0)
            cam_X = timath.normalize(timath.cross(cam_Z, cam_Y))
            cam_Y = timath.normalize(timath.cross(cam_X, cam_Z))

            for x, y in ti.ndrange(VIS_RESOLUTION[0], VIS_RESOLUTION[1]):
                # Compute x and y ray directions in neutral camera position
                rx = DOMAIN_SIZE_MAX * (ti.cast(x, PPTypes.FLOAT_GPU) / ti.cast(VIS_RESOLUTION[0], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
                ry = DOMAIN_SIZE_MAX * (ti.cast(y, PPTypes.FLOAT_GPU) / ti.cast(VIS_RESOLUTION[1], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
                ry /= aspect_ratio

                # Initialize ray origin and direction
                screen_pos = camera_pos + rx * cam_X + ry * cam_Y + screen_distance * cam_Z
                ray_dir = timath.normalize(screen_pos - camera_pos)

                # Get intersection of the ray with the volume AABB
                t = self.ray_AABB_intersection(
                    camera_pos,
                    ray_dir,
                    PPTypes.VEC3f(DOMAIN_MIN),
                    PPTypes.VEC3f(DOMAIN_MAX))
                ray_L = PPTypes.VEC3f(0.0, 0.0, 0.0)
                ray_delta = 1.71 * DOMAIN_SIZE_MAX / n_ray_steps_f

                # Check if we intersect the volume AABB at all
                if t[1] >= 0.0:
                    t[0] += RAY_EPSILON
                    t[1] -= RAY_EPSILON
                    t_current = t[0] + ti.random(dtype=PPTypes.FLOAT_GPU) * ray_delta
                    ray_pos = camera_pos + t_current * ray_dir

                    # Main integration loop
                    for i in ti.ndrange(n_steps):
                        if t_current >= t[1]:
                            break
                        trace_val = trace_field[self.world_to_grid_3D(
                            ray_pos,
                            PPTypes.VEC3f(DOMAIN_MIN),
                            PPTypes.VEC3f(DOMAIN_MAX),
                            PPTypes.VEC3i(TRACE_RESOLUTION))][0]
                        deposit_val = deposit_field[self.world_to_grid_3D(
                            ray_pos,
                            PPTypes.VEC3f(DOMAIN_MIN),
                            PPTypes.VEC3f(DOMAIN_MAX),
                            PPTypes.VEC3i(DEPOSIT_RESOLUTION))][current_deposit_index]
                        ray_L += PPTypes.VEC3f(
                            trace_vis * trace_val,
                            deposit_vis * deposit_val,
                            ti.pow(ti.log(1.0 + 0.2 * trace_vis * trace_val), 3.0)) / n_ray_steps_f
                        ray_pos += ray_delta * ray_dir
                        t_current += ray_delta

                # vis_field[x, y] = timath.pow(ray_L, 1.0/2.2)
                vis_field[x, y] = ((ray_L[1] + ray_L[0]) * self.TexSamplePosition(ray_L[0], colormap, 50)) * 0.5
            return



    ### 
    ### PATH TRACER CODE BEGINS HERE
    ### 

    @ti.kernel
    def render_visualization_3D_pathtraced(
        self,
        camera_distance: PPTypes.FLOAT_GPU,
        camera_polar: PPTypes.FLOAT_GPU,
        camera_azimuth: PPTypes.FLOAT_GPU,
        current_deposit_index: PPTypes.INT_GPU,
        accumulate_frame: PPTypes.INT_GPU,
        accumulation_count: PPTypes.INT_GPU,
        throughput_multiplier: PPTypes.FLOAT_GPU,
        num_samples: PPTypes.INT_GPU,
        max_bounces: PPTypes.INT_GPU,
        deposit_vis: PPTypes.FLOAT_GPU,
        trace_vis: PPTypes.FLOAT_GPU,
        TRACE_RESOLUTION: PPTypes.VEC3i,
        DEPOSIT_RESOLUTION: PPTypes.VEC3i,
        VIS_RESOLUTION: PPTypes.VEC2i,
        DOMAIN_SIZE_MAX: PPTypes.FLOAT_GPU,
        DOMAIN_MIN: PPTypes.VEC3f,
        DOMAIN_MAX: PPTypes.VEC3f,
        DOMAIN_CENTER: PPTypes.VEC3f,
        RAY_EPSILON: PPTypes.FLOAT_GPU,
        deposit_field: ti.template(),
        trace_field: ti.template(),
        scattering_anisotropy: PPTypes.FLOAT_GPU,
        sigma_e: PPTypes.FLOAT_GPU,
        sigma_a: PPTypes.FLOAT_GPU,
        sigma_s: PPTypes.FLOAT_GPU,
        sigma_t: PPTypes.FLOAT_GPU,
        trace_max: PPTypes.FLOAT_GPU,
        debug_mode: PPTypes.INT_GPU,
        vis_field: ti.template(),
        colormap: ti.template()):

            ## INITIAL CONSTANTS
            # the aspect ratio of the image
            aspect_ratio = ti.cast(VIS_RESOLUTION[0], PPTypes.FLOAT_GPU) / \
                ti.cast(VIS_RESOLUTION[1], PPTypes.FLOAT_GPU)
            screen_distance = DOMAIN_SIZE_MAX  # the distance of the screen from the camera
            camera_offset = camera_distance * PPTypes.VEC3f(ti.cos(camera_azimuth) * ti.sin(camera_polar), ti.sin(camera_azimuth) * ti.sin(
                camera_polar), ti.cos(camera_polar))  # the offset of the camera from the center of the volume
            camera_pos = DOMAIN_CENTER + camera_offset  # the position of the camera
            cam_Z = timath.normalize(-camera_offset)  # the camera's z axis
            cam_Y = PPTypes.VEC3f(0.0, 0.0, 1.0)
            cam_X = timath.normalize(timath.cross(cam_Z, cam_Y))  # the camera's x axis
            cam_Y = timath.normalize(timath.cross(cam_X, cam_Z))  # the camera's y axis
            m_b = -999.0
            new_sigma_a = 0.0
            new_sigma_s = 0.0
            albedo = 0.0
            if (sigma_t > 1.e-5):
                albedo = sigma_s / sigma_t
                new_sigma_a = (1.0 - albedo) * sigma_t
                new_sigma_s = albedo * sigma_t
            for x, y in ti.ndrange(VIS_RESOLUTION[0], VIS_RESOLUTION[1]):
                ###
                ### FOR EACH PIXEL ...
                ###

                ## Compute x and y ray directions in neutral camera position
                # x coordinate of the ray in the camera's coordinate system
                rx = DOMAIN_SIZE_MAX * \
                    (ti.cast(x, PPTypes.FLOAT_GPU) /
                     ti.cast(VIS_RESOLUTION[0], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
                # y coordinate of the ray in the camera's coordinate system
                ry = DOMAIN_SIZE_MAX * \
                    (ti.cast(y, PPTypes.FLOAT_GPU) /
                     ti.cast(VIS_RESOLUTION[1], PPTypes.FLOAT_GPU)) - 0.5 * DOMAIN_SIZE_MAX
                # correct for aspect ratio (+ resize handler, since this is computed per-frame)
                ry /= aspect_ratio

                ## Initialize ray origin and direction
                screen_pos = camera_pos + rx * cam_X + ry * cam_Y + screen_distance * \
                    cam_Z  # position of the ray in the world coordinate system

                # Great! Now we have everything we need for the ray for this pixel.
                rho_max_inv = 1.0 / self.trace_to_rho(trace_max)

                ## Get intersection of the ray with the volume AABB
                # the path radiance of the ray (total light it has accumulated)
                path_L = PPTypes.VEC3f(0.0, 0.0, 0.0)

                for s in range(num_samples):
                    ray_pos = camera_pos  # the position of the ray in the world coordinate system
                    # the direction of the ray in the world coordinate system
                    ray_dir = timath.normalize(screen_pos - ray_pos)

                    t = self.ray_AABB_intersection(ray_pos, ray_dir, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(
                        DOMAIN_MAX))  # returns [near, far] of volume intersection

                    if (t.y >= 0):
                        t.x += RAY_EPSILON
                        t.y -= RAY_EPSILON
                        ray_pos += t.x * ray_dir  # move the ray to the intersection point
                        # shorten the ray to the intersection interval
                        ray_dir = ray_dir * (t.y - t.x)
                        ray_dir = timath.normalize(ray_dir)  # normalize the ray direction
                        t_event = 0.0
                        throughput = 1.0

                        for n in range(max_bounces):
                            t = self.ray_AABB_intersection(ray_pos, ray_dir, PPTypes.VEC3f(DOMAIN_MIN), PPTypes.VEC3f(
                                DOMAIN_MAX))  # returns [near, far] of volume intersection
                            t_event = self.delta_tracking(ray_pos, ray_dir, 0.0, t.y, rho_max_inv, new_sigma_a, new_sigma_s, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis, deposit_vis)
                            if (t_event >= t.y):
                                break

                            m_b = sigma_t

                            ray_pos += t_event * ray_dir  # move the ray to the intersection point

                            sample = self.sample_volume(ray_pos, DEPOSIT_RESOLUTION, deposit_field, trace_field, current_deposit_index, DOMAIN_MIN, DOMAIN_MAX, TRACE_RESOLUTION, sigma_e, trace_vis, deposit_vis)
                            # combine the trace and deposit values
                            rho_event = sample[0] + sample[1]

                            # sample the volume at the intersection point
                            rho_event = rho_event * throughput_multiplier

                            emission = self.get_emitted_trace_L(rho_event, colormap)

                            if debug_mode == 1:
                                path_L = n
                            else:
                                path_L += throughput * rho_event * sigma_e * emission / 255

                            ray_dir = timath.normalize(
                                self.sample_HG(ray_dir, scattering_anisotropy))  # sample HGF

                            # if (_USE_RUSSIAN_ROULETTE == True):
                            #     if (n >= _RUSSIAN_ROULETTE_START_ORDER and ti.random() > albedo):
                            #         break
                            #     else:
                            #         throughput *= albedo
                            # else:
                            #     throughput *= albedo
                            throughput *= albedo #   <-- remove this line for russian roulette

                if (accumulate_frame == True):
                    if (path_L[0] != 0 and path_L[1] != 0 and path_L[2] != 0):
                        new_value = path_L / num_samples
                        old_value = vis_field[x, y]

                        weight = 1.0 / (accumulation_count + 1)
                        accumulatedPixel = old_value * (1 - weight) + new_value * weight
                        # average the path radiance between this and the previous frame
                        vis_field[x, y] = accumulatedPixel
                else:
                    # average the path radiance over the number of samples
                    vis_field[x, y] = path_L / num_samples
                    # TODO: Should switch to the ray marcher in this case ... don't trigger from here, instead from the handler class
            # print("aaa:", m_b) # <-- PRINT VIABLE HERE
            return
