# rs2_physics_core_enhanced_executable.py
# Paste this entire block into one Google Colab cell and run it.

"""
RS2 ENHANCED PHYSICS CORE ENGINE (EXECUTABLE)
Google Colab-ready with Interactive Widgets and Go system analysis features.
"""

import numpy as np
import math
import random
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import IPython.display as display
import ipywidgets as widgets

# ===================================================================
# ENHANCED PARAMETER SYSTEM (from Go implementation)
# ===================================================================

@dataclass
class SimulationParameters:
    """Configurable simulation parameters with presets"""
    grid_size: int = 10
    progression_damping_3d: float = 0.99
    gravity_strength_3d: float = 0.02
    quantum_chance_3d: float = 0.01
    vortex_strength_3d: float = 0.08
    cosmic_energy_strength: float = 0.01
    min_vortex_size: int = 3
    natural_to_atomic_mass: float = 0.99970644
    analysis_interval: int = 5
    max_simulation_ticks: int = 50
    initial_perturbations: int = 25
    perturbation_strength: float = 0.4
    spatial_variation: float = 0.2
    parameter_set_name: str = "Python_Pattern_Optimized_v1"
    last_modified: str = None
    
    def __post_init__(self):
        if self.last_modified is None:
            self.last_modified = datetime.now().isoformat()
    
    def to_dict(self):
        return self.__dict__

# Parameter presets (from Go implementation)
PARAMETER_PRESETS = {
    "python_optimized": SimulationParameters(
        grid_size=10, progression_damping_3d=0.99, gravity_strength_3d=0.02, 
        quantum_chance_3d=0.01, vortex_strength_3d=0.08, cosmic_energy_strength=0.01, 
        min_vortex_size=3, natural_to_atomic_mass=0.99970644, analysis_interval=5, 
        max_simulation_ticks=50, initial_perturbations=25, perturbation_strength=0.4, 
        spatial_variation=0.2, parameter_set_name="Python_Optimized"
    ),
    "gentle_formation": SimulationParameters(
        grid_size=10, progression_damping_3d=0.98, gravity_strength_3d=0.01, 
        quantum_chance_3d=0.02, vortex_strength_3d=0.05, cosmic_energy_strength=0.02, 
        min_vortex_size=2, natural_to_atomic_mass=0.99970644, analysis_interval=5, 
        max_simulation_ticks=50, initial_perturbations=50, perturbation_strength=0.3, 
        spatial_variation=0.1, parameter_set_name="Gentle_Formation"
    ),
    "strong_vortices": SimulationParameters(
        grid_size=10, progression_damping_3d=0.995, gravity_strength_3d=0.03, 
        quantum_chance_3d=0.005, vortex_strength_3d=0.12, cosmic_energy_strength=0.005, 
        min_vortex_size=4, natural_to_atomic_mass=0.99970644, analysis_interval=5, 
        max_simulation_ticks=50, initial_perturbations=15, perturbation_strength=0.6, 
        spatial_variation=0.3, parameter_set_name="Strong_Vortices"
    )
}

# ===================================================================
# ENHANCED PHYSICS STRUCTURES (from Go implementation)
# ===================================================================

class EnergyClass(Enum):
    HIGH = "High"
    MEDIUM = "Medium" 
    LOW = "Low"

class VortexType(Enum):
    STRONG_3D = "Strong 3D Vortex"
    MEDIUM_3D = "Medium 3D Vortex"
    WEAK_3D = "Weak 3D Vortex"

@dataclass
class Vortex3D:
    """Enhanced vortex structure with Go system analysis"""
    id: int
    position: Tuple[int, int, int]
    center: Tuple[float, float, float]
    strength: float
    vortex_type: VortexType
    size: int
    mass: float
    energy: float
    stability: float
    materiality: float
    cells_count: int
    creation_tick: int
    energy_class: EnergyClass
    atomic_element: str = ""
    calculated_mass: float = 0.0
    mass_accuracy: float = 0.0
    displacement: Tuple[int, int, int] = (0, 0, 0)
    rs2_notation: str = ""
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['vortex_type'] = self.vortex_type.value
        d['energy_class'] = self.energy_class.value
        return d

@dataclass
class VortexAnalysis:
    """Enhanced vortex analysis from Go system"""
    tick: int
    total_vortices: int
    total_energy: float
    vortex_details: List[Vortex3D]
    stability_avg: float
    materiality_avg: float
    energy_classes: Dict[str, int]
    
    def to_dict(self):
        d = self.__dict__.copy()
        d['vortex_details'] = [v.to_dict() for v in self.vortex_details]
        return d

@dataclass
class StabilitySummary:
    """Stability analysis summary from Go system"""
    vortex_count_stability: float
    energy_decay_rate: float
    materiality_balance: float
    particle_distribution: str
    python_match_score: float
    
    def to_dict(self):
        return self.__dict__

# ===================================================================
# ENHANCED CORE PHYSICS CLASSES
# ===================================================================

class EnhancedCell3D:
    """Enhanced cell with Go system features"""
    def __init__(self, x, y, z):
        self.position = (x, y, z)
        self.s = 1.0 + (random.random() * 0.4 - 0.2) 
        self.t = 1.0 
        self.rotation = [0.0, 0.0, 0.0]
        self.angular_velocity = [
            random.random() * 0.2 - 0.1,
            random.random() * 0.2 - 0.1, 
            random.random() * 0.2 - 0.1
        ]
        self.ratio_history = []
        self.dimensionality = 3
        self.materiality = 0.5
        
    def ratio(self):
        if self.t != 0:
            return self.s / self.t
        return 1.0
    
    def update_materiality(self):
        ratio = self.ratio()
        if ratio <= 1.0:
            self.materiality = 0.5 * (1.0 + ratio)
        else:
            self.materiality = 0.5 * (2.0 - 1.0/ratio)
    
    def angular_velocity_magnitude(self):
        return math.sqrt(
            self.angular_velocity[0]**2 + 
            self.angular_velocity[1]**2 + 
            self.angular_velocity[2]**2
        )
    
    def get_energy(self):
        return self.angular_velocity_magnitude() * self.s
    
    def progress(self, damping=0.97):
        """Progress cell state with damping logic from Go system"""
        self.t += 1
        
        # Update rotation
        for i in range(3):
            self.rotation[i] += self.angular_velocity[i]
        
        # Apply damping based on angular speed
        angular_speed = self.angular_velocity_magnitude()
        damping_factor = damping - (0.03 * math.exp(-angular_speed))
        for i in range(3):
            self.angular_velocity[i] *= damping_factor
        
        # Clamp values
        self.s = max(0.1, min(5.0, self.s))
        for i in range(3):
            self.angular_velocity[i] = max(-2.0, min(2.0, self.angular_velocity[i]))
        
        self.update_materiality()
        
        # Update ratio history
        self.ratio_history.append(self.ratio())
        if len(self.ratio_history) > 10:
            self.ratio_history = self.ratio_history[1:]

class EnhancedUniverse3D:
    """Enhanced universe with Go system parameter control and analysis"""
    
    def __init__(self, parameters: SimulationParameters):
        self.params = parameters
        self.width = parameters.grid_size
        self.height = parameters.grid_size  
        self.depth = parameters.grid_size
        self.tick_count = 0
        self.total_energy = 0.0
        self.materiality_avg = 0.5
        
        # Initialize grid
        self.material_grid = np.empty((self.depth, self.height, self.width), dtype=object)
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    self.material_grid[z][y][x] = EnhancedCell3D(x, y, z)
        
        self.vortices = []
        self.vortex_history = []
        
        self._add_3d_seed_perturbations()
    
    def _add_3d_seed_perturbations(self):
        """Add initial perturbations (from Go system)"""
        for _ in range(self.params.initial_perturbations):
            z = random.randint(0, self.depth - 1)
            y = random.randint(0, self.height - 1)
            x = random.randint(0, self.width - 1)
            cell = self.material_grid[z][y][x]
            
            energy_scale = self.params.perturbation_strength
            for i in range(3):
                cell.angular_velocity[i] += (random.random() * 2.0 - 1.0) * energy_scale
            
            cell.s += (random.random() * 2.0 - 1.0) * self.params.spatial_variation
            cell.update_materiality()
    
    def get_3d_neighbors(self, z, y, x):
        """Get 3D neighbors (26-connected)"""
        neighbors = []
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    nz, ny, nx = z + dz, y + dy, x + dx
                    if (0 <= nz < self.depth and 0 <= ny < self.height and 0 <= nx < self.width):
                        neighbors.append((nz, ny, nx))
        return neighbors
    
    def progress_3d_enhanced(self):
        """Enhanced progression (from Go system)"""
        self.total_energy = 0.0
        materiality_sum = 0.0
        cell_count = 0
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    cell = self.material_grid[z][y][x]
                    cell.progress(self.params.progression_damping_3d)
                    
                    materiality_sum += cell.materiality
                    cell_count += 1
                    self.total_energy += cell.angular_velocity_magnitude() * cell.s
        
        self.materiality_avg = materiality_sum / cell_count
        self.tick_count += 1
    
    def add_rotational_gravity_3d(self):
        """Apply rotational gravity (from Go system)"""
        # Only iterate over inner cells to safely check all 26 neighbors
        for z in range(1, self.depth - 1):
            for y in range(1, self.height - 1):
                for x in range(1, self.width - 1):
                    cell = self.material_grid[z][y][x]
                    neighbor_positions = self.get_3d_neighbors(z, y, x)
                    
                    avg_s = 0.0
                    avg_angular_vel = [0.0, 0.0, 0.0]
                    
                    for pos in neighbor_positions:
                        neighbor = self.material_grid[pos[0]][pos[1]][pos[2]]
                        avg_s += neighbor.s
                        for i in range(3):
                            avg_angular_vel[i] += neighbor.angular_velocity[i]
                    
                    count = len(neighbor_positions)
                    avg_s /= count
                    for i in range(3):
                        avg_angular_vel[i] /= count
                    
                    # Apply scalar gravity influence (S)
                    grad_s = avg_s - cell.s
                    cell.s += self.params.gravity_strength_3d * grad_s
                    
                    # Apply rotational gravity influence (Angular Velocity)
                    for i in range(3):
                        grad_rot = avg_angular_vel[i] - cell.angular_velocity[i]
                        cell.angular_velocity[i] += self.params.gravity_strength_3d * 0.3 * grad_rot
    
    def add_cosmic_energy_input_3d_enhanced(self):
        """Add cosmic energy input (from Go system)"""
        center_z = self.depth // 2
        center_y = self.height // 2
        center_x = self.width // 2
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    dist = math.sqrt(
                        (z - center_z)**2 + 
                        (y - center_y)**2 + 
                        (x - center_x)**2
                    )
                    
                    energy = self.params.cosmic_energy_strength * math.exp(-dist / 3.0)
                    cell = self.material_grid[z][y][x]
                    cell.s += energy
                    
                    # Quantum Chance Perturbation
                    if random.random() < self.params.quantum_chance_3d:
                        for i in range(3):
                            cell.angular_velocity[i] += energy * 0.5 * (random.random() - 0.5)
    
    def find_vortices_3d_enhanced(self):
        """Enhanced vortex detection (from Go system)"""
        visited = np.zeros((self.depth, self.height, self.width), dtype=bool)
        vortices = []
        vortex_id = 0
        
        for z in range(self.depth):
            for y in range(self.height):
                for x in range(self.width):
                    if not visited[z][y][x]:
                        cell = self.material_grid[z][y][x]
                        velocity_mag = cell.angular_velocity_magnitude()
                        
                        if velocity_mag > self.params.vortex_strength_3d:
                            vortex_cells, vortex_positions = self._flood_fill_vortex_3d(z, y, x, visited)
                            
                            if len(vortex_cells) >= self.params.min_vortex_size:
                                vortex = self._create_vortex_3d(vortex_id, vortex_cells, vortex_positions)
                                vortices.append(vortex)
                                vortex_id += 1
                        else:
                            visited[z][y][x] = True
        
        self.vortices = vortices
        return vortices
    
    def _flood_fill_vortex_3d(self, start_z, start_y, start_x, visited):
        """3D flood fill for vortex detection"""
        stack = [(start_z, start_y, start_x)]
        vortex_cells = []
        vortex_positions = []
        
        while stack:
            z, y, x = stack.pop()
            
            if (z < 0 or z >= self.depth or y < 0 or y >= self.height or 
                x < 0 or x >= self.width):
                continue

            if visited[z][y][x]:
                continue
            
            cell = self.material_grid[z][y][x]
            velocity_mag = cell.angular_velocity_magnitude()
            
            if velocity_mag > self.params.vortex_strength_3d:
                visited[z][y][x] = True
                vortex_cells.append(cell)
                vortex_positions.append((z, y, x))
                
                # Add neighbors
                neighbors = self.get_3d_neighbors(z, y, x)
                stack.extend(neighbors)
            else:
                visited[z][y][x] = True
        
        return vortex_cells, vortex_positions
    
    def _create_vortex_3d(self, vortex_id, cells, positions):
        """Create enhanced vortex analysis"""
        center = [0.0, 0.0, 0.0]
        total_strength = 0.0
        total_mass = 0.0
        total_stability = 0.0
        total_materiality = 0.0
        
        for i, (z, y, x) in enumerate(positions):
            cell = cells[i]
            strength = cell.angular_velocity_magnitude()
            stability = self._calculate_cell_stability(cell)
            
            # Position weighted by strength (Go system logic)
            center[0] += z * strength
            center[1] += y * strength
            center[2] += x * strength
            
            total_strength += strength
            total_mass += self._calculate_cell_mass(cell)
            total_stability += stability
            total_materiality += cell.materiality
        
        if total_strength > 0:
            for i in range(3):
                center[i] /= total_strength
        
        avg_strength = total_strength / len(cells)
        
        # Determine vortex type
        if avg_strength > 0.15:
            vortex_type = VortexType.STRONG_3D
        elif avg_strength > 0.08:
            vortex_type = VortexType.MEDIUM_3D
        else:
            vortex_type = VortexType.WEAK_3D
        
        # Calculate Energy Class
        vortex_energy = avg_strength * total_mass
        if vortex_energy >= 2.0:
            energy_class = EnergyClass.HIGH
        elif vortex_energy >= 0.5:
            energy_class = EnergyClass.MEDIUM
        else:
            energy_class = EnergyClass.LOW

        vortex = Vortex3D(
            id=vortex_id,
            position=(int(center[0]), int(center[1]), int(center[2])),
            center=tuple(center),
            strength=avg_strength,
            vortex_type=vortex_type,
            size=len(cells),
            mass=total_mass,
            energy=vortex_energy,
            stability=total_stability / len(cells),
            materiality=total_materiality / len(cells),
            cells_count=len(cells),
            creation_tick=self.tick_count,
            energy_class=energy_class
        )
        
        self._calculate_atomic_mass(vortex)
        
        return vortex
    
    def _calculate_cell_mass(self, cell):
        """Calculate cell mass (from Go system)"""
        base_mass = cell.angular_velocity_magnitude() * self.params.natural_to_atomic_mass
        dimension_scale = 1.0 + ((cell.dimensionality - 1) * 0.3)
        stability = self._calculate_cell_stability(cell)
        stability_bonus = 1.0 + (stability * 0.3)
        
        return base_mass * dimension_scale * stability_bonus
    
    def _calculate_cell_stability(self, cell):
        """Calculate cell stability from ratio history"""
        if len(cell.ratio_history) < 2:
            return 0.5
        
        ratios = np.array(cell.ratio_history)
        variance = np.var(ratios)
        
        stability = 1.0 / (1.0 + math.sqrt(variance) * 6.0)
        return max(0.0, min(1.0, stability))
    
    def _calculate_atomic_mass(self, vortex):
        """Calculate atomic mass and identify element"""
        base_mass = vortex.mass * self.params.natural_to_atomic_mass
        element, displacement = self._identify_element_from_vortex(vortex)
        corrected_mass = self._apply_displacement_correction(base_mass, displacement)
        
        # Calculate accuracy vs standard mass (using Hydrogen as reference)
        standard_mass = 1.00794
        accuracy = (1.0 - abs(corrected_mass - standard_mass) / standard_mass) * 100.0
        
        vortex.atomic_element = element
        vortex.calculated_mass = corrected_mass
        vortex.mass_accuracy = accuracy
        vortex.displacement = displacement
        vortex.rs2_notation = f"{displacement[0]}-{displacement[1]}-({displacement[2]})"
    
    def _identify_element_from_vortex(self, vortex):
        """Identify atomic element from vortex properties (from Go system)"""
        strength = vortex.strength
        size = vortex.size
        energy = vortex.energy
        stability = vortex.stability
        
        # NOTE: This logic is directly transcribed from the Go system
        if strength < 0.08 and size < 6 and energy < 0.5 and stability > 0.9:
            return "hydrogen", (2, 1, -1)
        elif strength < 0.12 and size < 10 and energy < 1.0 and stability > 0.85:
            return "helium", (2, 1, 0)
        elif strength < 0.18 and size < 15 and energy < 2.0 and stability > 0.8:
            return "lithium", (2, 1, 1)
        else:
            return "hydrogen", (2, 1, -1)
    
    def _apply_displacement_correction(self, base_mass, displacement):
        """Apply displacement correction to mass (from Go system)"""
        temporal_component = displacement[0] + displacement[1]
        spatial_component = abs(displacement[2])
        
        if displacement[2] < 0:
            spatial_component *= 0.8
        
        displacement_factor = 1.0 + (temporal_component * 0.1) - (spatial_component * 0.05)
        return base_mass * displacement_factor
    
    def run_enhanced_vortex_analysis(self):
        """Run enhanced vortex analysis (from Go system)"""
        vortices = self.find_vortices_3d_enhanced()
        
        energy_classes = {cls.value: 0 for cls in EnergyClass}
        total_stability = 0.0
        total_materiality = 0.0
        
        for vortex in vortices:
            total_stability += vortex.stability
            total_materiality += vortex.materiality
            energy_classes[vortex.energy_class.value] += 1
        
        stability_avg = total_stability / len(vortices) if vortices else 0.0
        materiality_avg = total_materiality / len(vortices) if vortices else 0.0
        
        analysis = VortexAnalysis(
            tick=self.tick_count,
            total_vortices=len(vortices),
            total_energy=self.total_energy,
            vortex_details=vortices,
            stability_avg=stability_avg,
            materiality_avg=materiality_avg,
            energy_classes=energy_classes
        )
        
        self.vortex_history.append(analysis)
        return analysis
    
    def _calculate_consistency(self, values):
        """Calculate consistency of values (from Go system)"""
        if len(values) < 2:
            return 0.0
        
        mean = np.mean(values)
        variance = np.var(values)
        return 1.0 / (1.0 + math.sqrt(variance))
    
    def _calculate_distribution_match(self):
        """Calculate distribution match with Python targets (from Go system)"""
        if not self.vortex_history:
            return 0.0
        
        last_analysis = self.vortex_history[-1]
        total = last_analysis.total_vortices
        if total == 0:
            return 0.0
        
        high_pct = last_analysis.energy_classes["High"] / total
        medium_pct = last_analysis.energy_classes["Medium"] / total
        low_pct = last_analysis.energy_classes["Low"] / total
        
        # Python Success Targets (transcribed from Go system)
        target_high = 0.08
        target_medium = 0.42
        target_low = 0.50
        
        match_score = 1.0 - (abs(high_pct - target_high) + 
                            abs(medium_pct - target_medium) + 
                            abs(low_pct - target_low)) / 3.0
        return max(0.0, match_score)
    
    def _get_particle_distribution_string(self):
        """Get particle distribution string"""
        if not self.vortex_history:
            return "No data"
        
        last_analysis = self.vortex_history[-1]
        return (f"High: {last_analysis.energy_classes['High']}, "
                f"Medium: {last_analysis.energy_classes['Medium']}, "
                f"Low: {last_analysis.energy_classes['Low']}")

    def calculate_stability_summary(self):
        """Calculate stability summary (from Go system)"""
        if len(self.vortex_history) < 2:
            return StabilitySummary(
                vortex_count_stability=0.0, energy_decay_rate=0.0, 
                materiality_balance=self.materiality_avg, 
                particle_distribution="Insufficient data", python_match_score=0.0
            )
        
        vortex_counts = [analysis.total_vortices for analysis in self.vortex_history]
        count_stability = self._calculate_consistency(vortex_counts)
        
        energy_start = self.vortex_history[0].total_energy
        energy_end = self.vortex_history[-1].total_energy
        energy_decay = (energy_start - energy_end) / energy_start if energy_start > 0 else 0.0
        
        distribution_score = self._calculate_distribution_match()
        
        # Python Match Score calculation (transcribed from Go system)
        python_match = (count_stability + (1.0 - energy_decay) + 
                        self.materiality_avg + distribution_score) / 4.0
        
        return StabilitySummary(
            vortex_count_stability=count_stability,
            energy_decay_rate=energy_decay,
            materiality_balance=self.materiality_avg,
            particle_distribution=self._get_particle_distribution_string(),
            python_match_score=python_match
        )
    
    def stream_enhanced_analysis_results(self, analysis, stability):
        """Stream analysis results in Go system format"""
        result = f"--- VORTEX ANALYSIS (Tick {self.tick_count}) ---\n"
        result += f"Found {analysis.total_vortices} vortices:\n"
        
        for i, vortex in enumerate(analysis.vortex_details[:3]): 
            result += (f"{i+1}. {vortex.vortex_type.value}(id={vortex.id}, size={vortex.size}, "
                      f"energy={vortex.energy:.3f}, stability={vortex.stability:.3f})\n")
            result += (f"   Element: {vortex.atomic_element}, Mass: {vortex.calculated_mass:.4f}, "
                      f"Accuracy: {vortex.mass_accuracy:.1f}%\n")
        
        result += f"\n=== STABILITY ANALYSIS ===\n"
        result += f"Vortex Count Stability: {stability.vortex_count_stability:.3f}\n"
        result += f"Energy Decay Rate: {stability.energy_decay_rate:.3f}\n"
        result += f"Materiality Balance: {stability.materiality_balance:.3f}\n"
        result += f"Particle Distribution: {stability.particle_distribution}\n"
        result += f"Python Match Score: {stability.python_match_score * 100:.1f}%\n"
        
        return result

# ===================================================================
# ENHANCED SIMULATION CONTROLLER
# ===================================================================

class Enhanced3DController:
    """Enhanced controller with Go system features"""
    
    def __init__(self, parameters: SimulationParameters = None):
        if parameters is None:
            parameters = PARAMETER_PRESETS["python_optimized"]
        
        self.universe = EnhancedUniverse3D(parameters)
        self.is_running = False
        self.params = parameters
        self.stability_history = []
    
    def run_enhanced_simulation(self, progress_callback=None):
        """Run enhanced simulation with Go system analysis"""
        self.is_running = True
        output_lines = []
        
        output_lines.append("=== ENHANCED 3D RECIPROCAL UNIVERSE SIMULATION ===")
        output_lines.append(f"Parameter Set: {self.params.parameter_set_name}")
        output_lines.append(f"Universe: {self.universe.width}x{self.universe.height}x{self.universe.depth} | Max Ticks: {self.params.max_simulation_ticks}")
        output_lines.append("=== ACTIVE PARAMETERS ===")
        output_lines.append(f"Gravity Strength: {self.params.gravity_strength_3d:.3f}")
        output_lines.append(f"Vortex Strength: {self.params.vortex_strength_3d:.3f}")
        output_lines.append(f"Cosmic Energy: {self.params.cosmic_energy_strength:.3f}")
        output_lines.append("")
        
        for tick in range(1, self.params.max_simulation_ticks + 1):
            if not self.is_running:
                break
            
            # Core Physics Progression (transcribed from Go system's run loop)
            self.universe.progress_3d_enhanced()
            self.universe.add_rotational_gravity_3d()
            self.universe.add_cosmic_energy_input_3d_enhanced()
            
            if tick % self.params.analysis_interval == 0 or tick == 1:
                analysis = self.universe.run_enhanced_vortex_analysis()
                stability = self.universe.calculate_stability_summary()
                self.stability_history.append(stability)
                
                analysis_output = self.universe.stream_enhanced_analysis_results(analysis, stability)
                output_lines.append(analysis_output)
                
                if progress_callback:
                    progress_callback({
                        'tick': tick, 'vortices': analysis.total_vortices, 
                        'match': stability.python_match_score
                    })
                
                if tick == 1 or tick == 26 or tick == 50:
                    output_lines.append(f"=== PROGRESS REPORT (Tick {tick}) ===")
                    output_lines.append(f"Vortices: {analysis.total_vortices} | Energy: {analysis.total_energy:.3f} | Python Match: {stability.python_match_score * 100:.1f}%")
            
            # Small delay for visualization/real-time feel
            time.sleep(0.001) # Very small delay for Colab (0.1 in Go was too slow)
        
        # Final analysis
        output_lines.append("\n=== FINAL STABILITY ANALYSIS ===")
        final_stability = self.universe.calculate_stability_summary()
        output_lines.append(f"Final Python Match Score: {final_stability.python_match_score * 100:.1f}%")
        output_lines.append(f"Final Vortex Count: {len(self.universe.vortices)}")
        output_lines.append(f"Final Energy: {self.universe.total_energy:.3f}")
        output_lines.append(f"Final Materiality: {self.universe.materiality_avg:.3f}")
        
        output_lines.append("=== ENHANCED 3D SIMULATION COMPLETE ===")
        self.is_running = False
        
        return output_lines
    
    def pause_simulation(self):
        self.is_running = False
    
    def reset_simulation(self, new_parameters: SimulationParameters = None):
        self.is_running = False
        if new_parameters:
            self.params = new_parameters
        self.universe = EnhancedUniverse3D(self.params)
        self.stability_history = []

# ===================================================================
# GOOGLE COLAB INTERFACE
# ===================================================================

# Global controller instance
controller = Enhanced3DController(PARAMETER_PRESETS["python_optimized"])

def create_colab_interface(controller: Enhanced3DController):
    """Creates a simple, interactive Colab interface."""
    
    # 1. Parameter Sliders (Main Tuning)
    grid_size_slider = widgets.IntSlider(value=controller.params.grid_size, min=5, max=15, step=1, description='Grid Size:', layout=widgets.Layout(width='90%'))
    gravity_slider = widgets.FloatSlider(value=controller.params.gravity_strength_3d, min=0.001, max=0.05, step=0.001, description='Gravity:', readout_format='.3f', layout=widgets.Layout(width='90%'))
    vortex_slider = widgets.FloatSlider(value=controller.params.vortex_strength_3d, min=0.05, max=0.2, step=0.01, description='Vortex Strength:', readout_format='.3f', layout=widgets.Layout(width='90%'))
    cosmic_slider = widgets.FloatSlider(value=controller.params.cosmic_energy_strength, min=0.005, max=0.03, step=0.001, description='Cosmic Energy:', readout_format='.3f', layout=widgets.Layout(width='90%'))

    # 2. Controls and Preset
    preset_dropdown = widgets.Dropdown(options=list(PARAMETER_PRESETS.keys()), value='python_optimized', description='Preset:', layout=widgets.Layout(width='250px'))
    run_button = widgets.Button(description="🚀 Run Simulation", button_style='success')
    pause_button = widgets.Button(description="⏸️ Pause", button_style='warning') 
    reset_button = widgets.Button(description="🔄 Reset", button_style='info')

    # 3. Output Area
    output_area = widgets.Output()
    status_label = widgets.Label(value="Status: Ready")

    def update_params_from_widgets():
        """Updates controller parameters from current widget values."""
        controller.params.grid_size = grid_size_slider.value
        controller.params.gravity_strength_3d = gravity_slider.value
        controller.params.vortex_strength_3d = vortex_slider.value
        controller.params.cosmic_energy_strength = cosmic_slider.value
        controller.params.parameter_set_name = "Custom_Colab_Tuning"

    def on_preset_change(change):
        """Updates sliders when a preset is selected."""
        preset_name = change['new']
        new_params = PARAMETER_PRESETS[preset_name]
        grid_size_slider.value = new_params.grid_size
        gravity_slider.value = new_params.gravity_strength_3d
        vortex_slider.value = new_params.vortex_strength_3d
        cosmic_slider.value = new_params.cosmic_energy_strength
        status_label.value = f"Status: Loaded preset '{preset_name}'"

    def progress_callback(data):
        """Updates the status label during the simulation run."""
        status_label.value = f"Status: Tick {data['tick']} | Vortices: {data['vortices']} | Python Match: {data['match']*100:.1f}%"

    def on_run_clicked(b):
        """Main run handler."""
        if controller.is_running:
            return
        
        with output_area:
            output_area.clear_output()
            print("--- Simulation Started ---")
            
            # 1. Update and Reset
            update_params_from_widgets()
            controller.reset_simulation(controller.params)
            status_label.value = "Status: Running..."

            # 2. Run
            results = controller.run_enhanced_simulation(progress_callback)
            
            # 3. Output
            for line in results:
                print(line)
            
            # 4. Final Status
            status_label.value = "Status: Complete"
            print("--- Simulation Complete ---")

    def on_pause_clicked(b):
        controller.pause_simulation()
        status_label.value = "Status: Paused"

    def on_reset_clicked(b):
        controller.reset_simulation()
        with output_area:
            output_area.clear_output()
            print("--- Simulation Reset ---")
        status_label.value = "Status: Ready"

    preset_dropdown.observe(on_preset_change, names='value')
    run_button.on_click(on_run_clicked)
    pause_button.on_click(on_pause_clicked)
    reset_button.on_click(on_reset_clicked)

    # Layout Construction
    header = widgets.HTML("<h3>RS2 Enhanced 3D - Parameter Control System (Colab)</h3>")
    param_controls = widgets.VBox([
        grid_size_slider, gravity_slider, vortex_slider, cosmic_slider
    ], layout=widgets.Layout(border='1px solid lightgray', padding='10px'))

    action_controls = widgets.HBox([
        preset_dropdown, run_button, pause_button, reset_button
    ])

    return widgets.VBox([
        header,
        param_controls,
        status_label,
        action_controls,
        widgets.HTML("<b>Simulation Output (Scrollable/Copyable)</b>"),
        output_area
    ])

# Execute the interface creation and display it
display.display(create_colab_interface(controller))
