import os 
from pulser import DigitalAnalogDevice, MockDevice, AnalogDevice

# Constants
N_QUBITS = 10
MAX_SAMPLES = 200
REGISTER_DIM = 30 # X*X μm dimension of qubits
SLIC_COMPACTNESS = 0.1

# Paths
DATA_ROOT = os.path.join(os.getcwd(), 'dataset')
NO_POLYP_DIR = os.path.join(DATA_ROOT, 'no_polyp')
POLYP_DIR = os.path.join(DATA_ROOT, 'polyp')


# Quantum backend settings
ODE_NSTEPS = 50000
ODE_NSTEPS_HIGH = 250000
MU_HYPERPARAMETER = 0.5
GLOBAL_PULSE_DURATION_COEF = 1

# Visualization settings
VISUALIZE_EXAMPLES = False

# Device
DEVICE = DigitalAnalogDevice

# Texture settings (homogeneity, contrast, dissimilarity, ASM, energy, correlation, pca)
TEXTURE_FEATURE = 'contrast'

# SVM class weights
CLASS_WEIGHTS = {
    0: 1.0, 
    1: 1.0
}

