import os 
from pulser import DigitalAnalogDevice, MockDevice, AnalogDevice

# Constants
N_QUBITS = 10
MAX_SAMPLES = 200
REGISTER_DIM = 20 # X*X μm dimension of qubits
SLIC_COMPACTNESS = 15

# Paths
# Data root from __file__
DATA_ROOT = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'pulser')
#DATA_ROOT = os.path.join(os.getcwd(), '..', '..', 'data', 'pulser')
NO_POLYP_DIR = os.path.join(DATA_ROOT, 'no_polyp')
POLYP_DIR = os.path.join(DATA_ROOT, 'polyp')


# Quantum backend settings
ODE_NSTEPS = 50000
ODE_NSTEPS_HIGH = 250000
MU_HYPERPARAMETER = 1.2
GLOBAL_PULSE_DURATION_COEF = 1

# Visualization settings
VISUALIZE_EXAMPLES = False

# Device
DEVICE = DigitalAnalogDevice

# Texture settings (homogeneity, contrast, dissimilarity, ASM, energy, correlation, pca)
TEXTURE_FEATURE = 'energy'

# SVM class weights
CLASS_WEIGHTS = {
    0: 1.0, 
    1: 1.0
}

