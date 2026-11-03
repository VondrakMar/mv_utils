"""Calculator wrapper that adds custom forces for the first few calculations."""
import numpy as np
from ase.calculators.calculator import Calculator, all_changes


class FirstStepOverride(Calculator):
    """Adds user-supplied forces for the first 10 calculations,
    then falls back to the wrapped calculator for everything after."""
    implemented_properties = ['energy', 'forces']

    def __init__(self, real_calc, custom_forces):
        super().__init__()
        self.real_calc = real_calc
        self.custom_forces = np.array(custom_forces)
        self.step_done = 0

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        # Always get real energy/forces so we have something sane for energy
        self.real_calc.calculate(atoms, properties=['energy', 'forces'],
                                 system_changes=system_changes)
        real_energy = self.real_calc.results['energy']
        real_forces = self.real_calc.results['forces']

        if self.step_done < 10:
            self.results['forces'] = real_forces + self.custom_forces.copy()
            self.results['energy'] = real_energy  # keep energy physical
            self.step_done += 1
        else:
            self.results['forces'] = real_forces
            self.results['energy'] = real_energy
