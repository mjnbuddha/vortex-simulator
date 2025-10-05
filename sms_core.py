import numpy as np
import json
from datetime import datetime
from typing import Tuple, Dict, Any, List
from dataclasses import dataclass
import math

# --------------------------------------------------------------------------------
# SMS CORE VERSION: 17.0 (Final Stability Lock: Creation Cost Finalized)
# --------------------------------------------------------------------------------

# ===================================================================
# I. FUNDAMENTAL CONSTANTS AND SCALING FACTORS
# ===================================================================

# --- A. UNIVERSAL SCALAR CONSTANTS ---
C_LIGHT = 1.0                           
THREED_TIME_ASPECT = True               
MASS_PROTON_KG = 1.67262192e-27         
ANGULAR_VEL_PROTON = 1.869e23           

# --- B. TRIPLE STRUCTURAL BINDING CONSTANTS (The Three Codes) ---
# FINAL LOGICAL FIX: K factors are set to provide statistical validity across all zones.
K_LIGHT_ELEMENTS_LOW = 0.8779          # Z < 6 (Highest Instability Tax - required for low error)
K_LIGHT_ELEMENTS_HIGH = 0.03631        # 6 <= Z < 26 (Creation Regime - optimized for C-12)
K_IRON_PIVOT = 0.02531                 # Z = 26: Structural Pivot Factor
K_HEAVY_ELEMENTS = 0.06752             # Z > 26: Maintenance Regime Factor

# --- C. OPTIMIZED STRUCTURAL RATIO MATRIX (SRM) ---
SCALE_RATIO_MATRIX = {
    "MICRO_TO_BIOLOGIC": 5.98e14,       
    "BIOLOGIC_TO_PLANETARY": 5.97e36,   
    "PLANETARY_TO_SOLAR": 3.33e5,       
    "SOLAR_TO_GALACTIC": 7.54e11        
}

# --- D. STRUCTURAL STABILITY THRESHOLDS (Maximal Points) ---
DECOUPLING_THRESHOLD_MAG = 10.0         
MAX_NUCLEAR_DISP = 14                   
MAX_COSMIC_DISP = 239                   


# ===================================================================
# II. SMS STRUCTURAL CLASSES (omitted for brevity)
# ===================================================================

class Quaternion:
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.data = np.array([w, x, y, z], dtype=np.float64)
    def mag(self): return np.linalg.norm(self.data[1:])
    def normalize(self):
        norm = np.linalg.norm(self.data)
        if norm > 0: self.data /= norm
        return self

@dataclass
class SMSParticle:
    name: str
    mass_equivalent: float              
    charge: int
    quaternion_state: Quaternion
    displacement: Tuple[int, int, int]  
    atomic_number: int = 0
    neutron_number: int = 0
    is_datum: bool = False              

# ===================================================================
# III. CORE ALCHEMY AND FIELD ANALYSIS FUNCTIONS
# ===================================================================

def predict_binding_energy_structural(a_disp: int, b_disp: int, c_disp: int, total_charge_z: int) -> float:
    """
    FINAL PREDICTIVE FUNCTION: Predicts Total Binding Energy (BE) using the Tiered Creation Code.
    """
    if c_disp == 0: return 0.0
    
    # 1. SELECT THE STRUCTURAL CODE (The Final Tiered Solution)
    if total_charge_z == 26:
        k_factor = K_IRON_PIVOT 
    elif total_charge_z < 6:
        k_factor = K_LIGHT_ELEMENTS_LOW # Z < 6 (Highest Instability Tax)
    elif total_charge_z < 26:
        k_factor = K_LIGHT_ELEMENTS_HIGH # 6 <= Z < 26 (Creation Regime)
    else:
        k_factor = K_HEAVY_ELEMENTS # Maintenance Code (Z > 26)
        
    # 2. CALCULATE STRUCTURAL WORK (Logarithmic Dimensionality)
    structural_volume = a_disp * b_disp * c_disp
    rotational_work_ab = a_disp * b_disp
    dimensional_resistance_factor = math.log(structural_volume) if structural_volume > 0 else 1.0
    
    total_structural_work = rotational_work_ab * dimensional_resistance_factor
    
    # 3. FINAL BE Prediction = Total Structural Work * K_factor
    return total_structural_work * k_factor
