import os 
from pulser import DigitalAnalogDevice, MockDevice, AnalogDevice

# Constants
N_QUBITS = 10
MAX_SAMPLES = 400
REGISTER_DIM = 5 # X*X μm dimension of qubits

# Paths
DATA_ROOT = os.path.join(os.getcwd(), 'dataset')
NO_POLYP_DIR = os.path.join(DATA_ROOT, 'no_polyp')
POLYP_DIR = os.path.join(DATA_ROOT, 'polyp')

# Quantum backend settings
ODE_NSTEPS = 50000
ODE_NSTEPS_HIGH = 250000

# Visualization settings
VISUALIZE_EXAMPLES = False

# Device
DEVICE = DigitalAnalogDevice

# Texture settings (homogeneity, contrast, dissimilarity, ASM, energy, correlation, pca)
TEXTURE_FEATURE = 'contrast'