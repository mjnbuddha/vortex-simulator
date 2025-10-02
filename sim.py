# rs2_particle_simulator.py
"""
RS2 PARTICLE SIMULATOR (Scalar Dynamics Kernel)
Integrates Quaternion motion, Peret's Mass Conversion, and Emergent Gravity
for building the Particle Zoo and Molecular Bonds.
"""

import numpy as np
import math
import random
import json
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

# ===================================================================
# I. CORE CONSTANTS AND SCALAR RECIPROCITY (Bruce Peret's Constants)
# ===================================================================

NATURAL_TO_ATOMIC_MASS = 0.99970644  # Peret's Mass Conversion Factor
GOLDEN_RATIO = 1.618033988749895
GRAVITY_COUPLING_FACTOR = 0.025      # Tuned factor for initial stability

# Dimensional thresholds determine classification
DIMENSIONAL_THRESHOLDS = {
    1: {'size': 2, 'energy_min': 0.0},  # Scalar/Linear
    2: {'size': 6, 'energy_min': 0.05}, # Electric/Planar (Complex)
    3: {'size': 8, 'energy_min': 0.1},  # Magnetic/Volumetric (Quaternion)
}

ROTATIONAL_ENERGY_FACTORS = {
    'quaternion_vector': 0.8,
}

# ===================================================================
# II. CORE PHYSICS CLASSES (Motion Identity and Agency)
# ===================================================================

class Quaternion:
    """Mathematical structure for 3D rotation and Spin (4D Atomic Zone)"""
    def __init__(self, w=1.0, x=0.0, y=0.0, z=0.0):
        self.data = np.array([w, x, y, z], dtype=np.float64)
        self.normalize()
    
    @property
    def w(self(self): return self.data[0] # Scalar part (Linear Progression)
    @property
    def x(self): return self.data[1] # i-component (Magnetic/Rotation 1)
    @property
    def y(self): return self.data[2] # j-component (Magnetic/Rotation 2)
    @property
    def z(self): return self.data[3] # k-component (Electric/Rotation 3)
    
    def norm(self): return np.linalg.norm(self.data)
    
    def normalize(self):
        norm = self.norm()
        if norm > 0: self.data /= norm
        return self

class RS2Cell:
    """Fundamental unit of scalar motion, carrying dimensional displacement"""
    def __init__(self, x, y, z):
        self.position = (x, y, z)
        self.s = 1.0  # Spatial magnitude
        self.t = 1.0  # Temporal magnitude
        # Quaternion represents the instantaneous spin state
        self.quaternion = Quaternion(1.0, 0.0, 0.0, 0.0) 
        self.history = []
        
    def progress(self, damping=0.97):
        """Advances cell state (time progression and natural damping)"""
        self.s = damping * self.s + (1 - damping) * 1.0
        self.t = damping * self.t + (1 - damping) * 1.0
        self.quaternion.normalize()
        
        # Record state for stability analysis
        self.history.append((self.s, self.t, self.quaternion.x, self.quaternion.y, self.quaternion.z))
        if len(self.history) > 10: self.history.pop(0)
    
    def get_rotational_energy(self):
        """Energy derived from Quaternion vector (rotational speed)"""
        vec_mag = math.sqrt(self.quaternion.x**2 + self.quaternion.y**2 + self.quaternion.z**2)
        return vec_mag * ROTATIONAL_ENERGY_FACTORS['quaternion_vector']
    
    def get_materiality(self):
        """Calculate material-cosmic balance (0.5 = Unity)"""
        ratio = self.s / self.t
        return 0.5 * (1.0 + ratio) if ratio <= 1.0 else 0.5 * (2.0 - 1.0/ratio)
    
    def calculate_stability(self):
        """Stability based on history consistency (Inverse of variance)"""
        if len(self.history) < 5: return 0.8
        recent_states = np.array(self.history)
        variability = np.std(recent_states, axis=0)
        return 1.0 / (1.0 + np.mean(variability))

class RS2Vortex:
    """Vortex structure representing potential particles (The Particle Zoo)"""
    def __init__(self, vortex_id, cells):
        self.id = vortex_id
        self.cells = cells
        self.size = len(cells)
        self.center = self._calculate_center()
        self.history = []
        
    def analyze(self, current_tick):
        """Calculate mass, stability, and dimensional class"""
        center = self._calculate_center()
        energy = np.mean([cell.get_rotational_energy() for cell in self.cells])
        materiality = np.mean([cell.get_materiality() for cell in self.cells])
        stability = np.mean([cell.calculate_stability() for cell in self.cells])
        
        # Dimensional Classification (Based on size and thresholds)
        dimension = self._determine_dimension()
        
        # Mass Calculation (Peret's Method - Cosmic balance optimization)
        cosmic_balance = 1.0 - materiality
        cosmic_energy = energy * (1.0 + cosmic_balance * GOLDEN_RATIO)
        mass = cosmic_energy * NATURAL_TO_ATOMIC_MASS # Final atomic mass
        
        vortex_data = {
            'id': self.id,
            'tick': current_tick,
            'mass': mass,
            'stability': stability,
            'dimension': dimension,
            'energy': energy,
            'size': self.size,
        }
        self.history.append(vortex_data)
        return vortex_data
    
    def _calculate_center(self):
        z_sum = sum(cell.position[0] for cell in self.cells)
        y_sum = sum(cell.position[1] for cell in self.cells)
        x_sum = sum(cell.position[2] for cell in self.cells)
        return (x_sum/self.size, y_sum/self.size, z_sum/self.size)
    
    def _determine_dimension(self):
        if self.size >= DIMENSIONAL_THRESHOLDS[3]['size']: return 3
        if self.size >= DIMENSIONAL_THRESHOLDS[2]['size']: return 2
        return 1

class RS2Universe:
    """Main simulation universe, manages progression and forces"""
    def __init__(self, width=15, height=15, depth=5):
        self.width, self.height, self.depth = width, height, depth
        self.grid = np.empty((depth, height, width), dtype=object)
        for z in range(depth):
            for y in range(height):
                for x in range(width):
                    self.grid[z, y, x] = RS2Cell(x, y, z)
        
        self.vortices = []
        self.tick_count = 0
        self.vortex_id_counter = 0

    def seed_perturbation(self, x, y, z, energy_scale=0.3):
        """Initial seeding for high-energy vortex formation"""
        if 0 <= x < self.width and 0 <= y < self.height and 0 <= z < self.depth:
            cell = self.grid[z, y, x]
            cell.s = 1.0 / energy_scale
            cell.t = 1.0 * energy_scale
            
            # Seed high rotational energy
            rot_scale = math.sqrt(energy_scale)
            real_part = math.sqrt(1.0 - energy_scale)
            
            # Initialize with non-zero vector components for immediate rotation
            vec = np.random.uniform(-1, 1, 3)
            cell.quaternion = Quaternion(real_part, *(vec / np.linalg.norm(vec) * rot_scale))

    def progress(self):
        """Advances simulation by one universal tick"""
        self._apply_gravity_and_rotation()
        self._quantum_fluctuations()
        
        for cell in self.grid.flatten():
            if cell: cell.progress()
        
        self._find_vortices()
        self.tick_count += 1

    def _apply_gravity_and_rotation(self):
        """Applies spatial distortion (gravity) and rotational coupling"""
        new_grid = np.copy(self.grid)
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.grid[z, y, x]
                    if not cell: continue
                    
                    # Calculate net influence from neighbors (Moore neighborhood)
                    s_influence_sum, q_influence_sum = 0.0, np.zeros(4)
                    neighbor_count = 0
                    
                    for dz in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            for dx in [-1, 0, 1]:
                                if dz == 0 and dy == 0 and dx == 0: continue
                                nz, ny, nx = z+dz, y+dy, x+dx
                                if 0 <= nz < self.depth and 0 <= ny < self.height and 0 <= nx < self.width:
                                    neighbor = self.grid[nz, ny, nx]
                                    if neighbor:
                                        # 1. Spatial Distortion (Gravity)
                                        s_influence_sum += neighbor.s - cell.s
                                        
                                        # 2. Rotational Coupling (Vortex Stabilization)
                                        # This strong coupling (factor 1.5) creates vortex stability
                                        q_influence_sum += (neighbor.quaternion.data - cell.quaternion.data) * 1.5 
                                        neighbor_count += 1
                                    
                    if neighbor_count > 0:
                        # Apply normalized spatial gradient
                        new_grid[z, y, x].s += GRAVITY_COUPLING_FACTOR * (s_influence_sum / neighbor_count)
                        
                        # Apply normalized rotational gradient
                        new_grid[z, y, x].quaternion.data += GRAVITY_COUPLING_FACTOR * (q_influence_sum / neighbor_count)

        self.grid = new_grid

    def _quantum_fluctuations(self):
        """Simulates spontaneous emergence/annihilation of motion (energy)"""
        for cell in self.grid.flatten():
            if random.random() < 0.005: # Low random chance for creation
                # Create: Adds random energy and rotation
                if random.random() < 0.5:
                    cell.s += random.uniform(1.0, 3.0) 
                else:
                    vec = np.random.uniform(-0.1, 0.1, 3)
                    cell.quaternion.data[1:] += vec
                    cell.quaternion.normalize()
    
    def _find_vortices(self, energy_threshold=0.01):
        """Detects connected cell structures above a rotational energy threshold"""
        visited = np.zeros((self.depth, self.height, self.width), dtype=bool)
        new_vortices = []
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    if not visited[z, y, x]:
                        cell = self.grid[z, y, x]
                        if cell.get_rotational_energy() > energy_threshold:
                            vortex_cells = self._flood_fill_vortex(z, y, x, visited, energy_threshold)
                            if len(vortex_cells) >= 3: # Minimum 3 cells for 2D/3D structure
                                vortex = RS2Vortex(self.vortex_id_counter, vortex_cells)
                                self.vortex_id_counter += 1
                                new_vortices.append(vortex)
        
        # NOTE: For long-term cataloging, we don't clear the vortex list.
        # The decay is tracked implicitly by subsequent _find_vortices runs finding fewer cells.
        self.vortices.extend(new_vortices)
    
    def _flood_fill_vortex(self, start_z, start_y, start_x, visited, energy_threshold):
        """Helper for finding contiguous vortex regions"""
        stack = [(start_z, start_y, start_x)]
        vortex_cells = []
        
        while stack:
            z, y, x = stack.pop()
            if not (0 <= z < self.depth and 0 <= y < self.height and 0 <= x < self.width) or visited[z, y, x]:
                continue
                
            cell = self.grid[z, y, x]
            if cell.get_rotational_energy() > energy_threshold:
                visited[z, y, x] = True
                vortex_cells.append(cell)
              
                for dz in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dz != 0 or dy != 0 or dx != 0:
                                stack.append((z+dz, y+dy, x+dx))
        return vortex_cells

# ===================================================================
# III. DATA EXPORT AND CATALOGING (The Particle Zoo)
# ===================================================================

def export_vortex_catalog(universe, filename="rs2_particle_catalog.json"):
    """Exports data for stable particles for external analysis and molecular bonding simulation"""
    
    catalog = []
    # Analyze all found vortices (including those that decayed)
    all_vortices = [v for v in universe.vortices if v.history]
    
    for vortex in all_vortices:
        latest = vortex.history[-1]
        
        # Filter for relatively stable particles (stability > 0.8) and non-trivial mass
        if latest['stability'] > 0.8 and latest['mass'] > 0.05:
            # Map Angular Velocity components to Spin Identity
            avg_q_vec = np.mean([c.quaternion.data[1:] for c in vortex.cells], axis=0)
            
            spin_identity = {
                'magnetic_i': avg_q_vec[0], # X-component
                'magnetic_j': avg_q_vec[1], # Y-component
                'electric_k': avg_q_vec[2], # Z-component
            }
            
            catalog.append({
                'Vortex_ID': latest['id'],
                'Mass_u': f"{latest['mass']:.6f}",
                'Stability': f"{latest['stability']:.3f}",
                'Dimensional_Class': latest['dimension'],
                'Size_Cells': latest['size'],
                'Spin_Identity': spin_identity,
                'Center': vortex.center
            })

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(catalog, f, indent=4)
        
    print(f"\n--- CATALOG EXPORT COMPLETE ---")
    print(f"Exported {len(catalog)} stable particles to {filename}")
    return catalog

# ===================================================================
# IV. SIMULATION CONTROLLER AND ENTRY POINT
# ===================================================================

def run_particle_zoo(ticks=50, export_file="rs2_particle_catalog.json"):
    """Initializes and runs the simulation, then exports the Particle Zoo"""
    
    universe = RS2Universe(width=15, height=15, depth=5)
    print("--- RS2 PARTICLE ZOO SIMULATOR ---")
    print(f"Initializing {universe.width}x{universe.height}x{universe.depth} grid...")
    
    # Seed specific initial perturbations for complex vortex formation
    seed_points = [
        (7, 7, 2, 0.4), # Center (high energy)
        (3, 10, 1, 0.2),
        (12, 4, 3, 0.3),
        (2, 2, 0, 0.1) # Low energy seed
    ]
    for x, y, z, energy in seed_points:
        universe.seed_perturbation(x, y, z, energy)
        
    print(f"Running simulation for {ticks} ticks...")
    
    # Main simulation loop
    for tick in range(ticks):
        universe.progress()
        if (tick + 1) % 5 == 0:
            print(f"Tick {tick+1}/{ticks}: Found {len(universe.vortices)} total vortices (Last 5 ticks)")

    # Analyze and export final catalog
    catalog = export_vortex_catalog(universe, filename=export_file)
    
    # Simple particle classification report
    classified_counts = defaultdict(int)
    for p in catalog:
        mass = float(p['Mass_u'])
        if mass > 4.0: classified_counts['Helium+'] += 1
        elif mass > 2.0: classified_counts['Deuterium/Tritium'] += 1
        elif mass > 0.8: classified_counts['Proton/Neutron'] += 1
        else: classified_counts['Light/EM'] += 1
        
    print("\n--- FINAL PARTICLE CLASSIFICATION REPORT ---")
    print(f"Total Stable Vortices Cataloged: {len(catalog)}")
    for class_name, count in classified_counts.items():
        print(f"- {class_name}: {count} stable structures found.")

if __name__ == "__main__":
    # Ensure NumPy is available for the Quaternion math
    try:
        np.array([1, 2, 3])
    except NameError:
        print("ERROR: NumPy not found. Please ensure NumPy is installed in your environment.")
        exit(1)
        
    # Run the Particle Zoo Simulation
    run_particle_zoo(ticks=75) 
