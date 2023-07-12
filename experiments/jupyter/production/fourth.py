import first 
import third
import taichi as ti
from numpy.random import default_rng
import numpy as np

## Default root directory
ROOT = '../../../'

## Data input file - leave empty for random set
# INPUT_FILE = ''
INPUT_FILE = ROOT + 'data/csv/sample_2D_linW.csv'
# INPUT_FILE = ROOT + 'data/csv/sample_2D_logW.csv'

## Initialize Taichi
ti.init(arch=ti.cpu)
rng = default_rng()

## Initialize data and agents
data = None
DOMAIN_MIN = None
DOMAIN_MAX = None
DOMAIN_SIZE = None
N_DATA = None
N_AGENTS = None
AVG_WEIGHT = 10.0

## Load data
## If no input file then generate a random dataset
if len(INPUT_FILE) > 0:
    data = np.loadtxt(INPUT_FILE, delimiter=",").astype(first.FLOAT_CPU)
    N_DATA = data.shape[0]
    N_AGENTS = third.N_AGENTS_DEFAULT
    domain_min = (np.min(data[:,0]), np.min(data[:,1]))
    domain_max = (np.max(data[:,0]), np.max(data[:,1]))
    domain_size = np.subtract(domain_max, domain_min)
    DOMAIN_MIN = (domain_min[0] - third.DOMAIN_MARGIN * domain_size[0], domain_min[1] - third.DOMAIN_MARGIN * domain_size[1])
    DOMAIN_MAX = (domain_max[0] + third.DOMAIN_MARGIN * domain_size[0], domain_max[1] + third.DOMAIN_MARGIN * domain_size[1])
    DOMAIN_SIZE = np.subtract(DOMAIN_MAX, DOMAIN_MIN)
    AVG_WEIGHT = np.mean(data[:,2])
else:
    N_DATA = third.N_DATA_DEFAULT
    N_AGENTS = third.N_AGENTS_DEFAULT
    DOMAIN_SIZE = third.DOMAIN_SIZE_DEFAULT
    DOMAIN_MIN = (0.0, 0.0)
    DOMAIN_MAX = third.DOMAIN_SIZE_DEFAULT
    data = np.zeros(shape=(N_DATA, 3), dtype = first.FLOAT_CPU)
    data[:, 0] = rng.normal(loc = DOMAIN_MIN[0] + 0.5 * DOMAIN_MAX[0], scale = 0.13 * DOMAIN_SIZE[0], size = N_DATA)
    data[:, 1] = rng.normal(loc = DOMAIN_MIN[1] + 0.5 * DOMAIN_MAX[1], scale = 0.13 * DOMAIN_SIZE[1], size = N_DATA)
    data[:, 2] = AVG_WEIGHT

## Derived constants
DATA_TO_AGENTS_RATIO = first.FLOAT_CPU(N_DATA) / first.FLOAT_CPU(N_AGENTS)
DOMAIN_SIZE_MAX = np.max([DOMAIN_SIZE[0], DOMAIN_SIZE[1]])
TRACE_RESOLUTION = first.INT_CPU((first.FLOAT_CPU(third.TRACE_RESOLUTION_MAX) * DOMAIN_SIZE[0] / DOMAIN_SIZE_MAX, first.FLOAT_CPU(third.TRACE_RESOLUTION_MAX) * DOMAIN_SIZE[1] / DOMAIN_SIZE_MAX))
DEPOSIT_RESOLUTION = (TRACE_RESOLUTION[0] // third.DEPOSIT_DOWNSCALING_FACTOR, TRACE_RESOLUTION[1] // third.DEPOSIT_DOWNSCALING_FACTOR)
VIS_RESOLUTION = TRACE_RESOLUTION

## Init agents
agents = np.zeros(shape=(N_AGENTS, 4), dtype = first.FLOAT_CPU)
agents[:, 0] = rng.uniform(low = DOMAIN_MIN[0] + 0.001, high = DOMAIN_MAX[0] - 0.001, size = N_AGENTS)
agents[:, 1] = rng.uniform(low = DOMAIN_MIN[1] + 0.001, high = DOMAIN_MAX[1] - 0.001, size = N_AGENTS)
agents[:, 2] = rng.uniform(low = 0.0, high = 2.0 * np.pi, size = N_AGENTS)
agents[:, 3] = 1.0

print('Simulation domain min:', DOMAIN_MIN)
print('Simulation domain max:', DOMAIN_MAX)
print('Simulation domain size:', DOMAIN_SIZE)
print('Trace grid resolution:', TRACE_RESOLUTION)
print('Deposit grid resolution:', DEPOSIT_RESOLUTION)
print('Data sample:', data[0, :])
print('Agent sample:', agents[0, :])
print('Number of agents:', N_AGENTS)
print('Number of data points:', N_DATA)

## Allocate GPU memory fields
## Keep in mind that the dimensions of these fields are important in the subsequent computations;
## that means if they change the GPU kernels and the associated handling code must be modified as well
data_field = ti.Vector.field(n = 3, dtype = first.FLOAT_GPU, shape = N_DATA)
agents_field = ti.Vector.field(n = 4, dtype = first.FLOAT_GPU, shape = N_AGENTS)
deposit_field = ti.Vector.field(n = 2, dtype = first.FLOAT_GPU, shape = DEPOSIT_RESOLUTION)
trace_field = ti.Vector.field(n = 1, dtype = first.FLOAT_GPU, shape = TRACE_RESOLUTION)
vis_field = ti.Vector.field(n = 3, dtype = first.FLOAT_GPU, shape = VIS_RESOLUTION)
print('Total GPU memory allocated:', first.INT_CPU(4 * (\
    data_field.shape[0] * 3 + \
    agents_field.shape[0] * 4 + \
    deposit_field.shape[0] * deposit_field.shape[1] * 2 + \
    trace_field.shape[0] * trace_field.shape[1] * 1 + \
    vis_field.shape[0] * vis_field.shape[1] * 3 \
    ) / 2 ** 20), 'MB')