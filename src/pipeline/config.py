import os 
from pulser import DigitalAnalogDevice, MockDevice, AnalogDevice

# Constants
N_QUBITS = 15
MAX_SAMPLES = 400
REGISTER_DIM = 30 # X*X μm dimension of qubits
SLIC_COMPACTNESS = 10

# Paths
DATA_ROOT = os.path.join(os.getcwd(), 'dataset')
NO_POLYP_DIR = os.path.join(DATA_ROOT, 'no_polyp')
POLYP_DIR = os.path.join(DATA_ROOT, 'polyp')

# Quantum backend settings
ODE_NSTEPS = 50000
ODE_NSTEPS_HIGH = 250000
MU_HYPERPARAMETER = 0.5

# Visualization settings
VISUALIZE_EXAMPLES = False

# Device
DEVICE = DigitalAnalogDevice

# Texture settings (homogeneity, contrast, dissimilarity, ASM, energy, correlation, pca)
TEXTURE_FEATURE = 'pca'

# SVM class weights
CLASS_WEIGHTS = 'balanced'