# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 12:05:26 2023

@author: naouf
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd



     # Conditions initiales
n = 2
F = 96500
Fe2_in = 0 
Fe3_in = 0 
HO_in = 0
HO2_in = 0
O2_minus_in = 0
SO4_2_in = 0 
SO4_minus_in= 0
FeSO4_in = 0
FeSO4_plus_in = 0
FeSO4_2_minus_in = 0
FeOH2_plus_in = 0
FeOH2_plus2_in = 0
FeHO2_2_plus_in = 0
Fe2OH2_4_plus_in = 0
FeOH_OH2_plus_in = 0
pH_in = 3

# Définition des constantes de vitesse
k1 = 6.3e4
k2 = 2.0e-6
k28 = 3.3e4
k47 = 1.58e5
k48 = 1.0e7
k3 = 3.2e5
k4 = 1.2e3
k5 = 3.6e2
k6 = 1.0e4
k7 = 5.0e4
k29 = 5.2e6
k30 = 8.3e2
k49 = 1.0e7
k37 = 7.1e6
k38 = 1.01e7
k31 = 9.7e4
k32 = 5.0e-4
k33 = 1.3e-4
k8 = 0  # Négligeable
k9 = 5.0e4
k10 = 2.29e8
k16 = 3.89e9
k17 = 4.47e7
k51 = 3.47e8
k39 = 1.4e4
k40 = 3.5e2
k45 = 3.0e5
k46 = 1.4e4
k34 = 1.2e4
k50 = 3.5e6
k11 = 3.0e5
k12 = 1.0e10
k18 = 1.0e10
k19 = 1.0e10
k52 = 1.0e10
k20 = 2.9e7
k21 = 7.62e3
k22 = 8.0e3
k13 = 2.0e-6
k23 = 1.0e7
k24 = 1.0e4
k25 = 1.0e4
k26 = 3.1e4
k27 = 1.0e7
k14 = 2.3e-3
k35 = 2.0e3
k36 = 1.0e7
k15 = 2.3e-3
                
H_plus_in = 10**(-pH_in) 
OH_minus_in = 10**(pH_in - 14)

# Définir les équations différentielles
def system_of_equations(t, y):
    dFe2_dt, dFe3_dt, dH2O2_dt, dHO_dt, dHO2_dt, dO2_minus_dt, dSO4_2_dt, dHSO4_dt, dO2_dt, dSO4_minus_dt, dFeSO4_dt, dFeSO4_plus_dt, dFeSO4_2_minus_dt, dFeOH2_plus_dt, dFeOH2_plus2_dt, dFeHO2_2_plus_dt, dFe2OH2_4_plus_dt, dFeOH_OH2_plus_dt, dH_plus_dt, dOH_minus_dt, dVM_dt, Fe2, Fe3, H2O2, HO, HO2, O2_minus, SO4_2, HSO4, O2, SO4_minus, FeSO4, FeSO4_plus, FeSO4_2_minus, FeOH2_plus, FeOH2_plus2, FeHO2_2_plus, Fe2OH2_4_plus, FeOH_OH2_plus, H_plus, OH_minus = y
    dFe2_dt = (ic * S) / (n * F * V) - k1 * Fe2 * H2O2 + k2 * Fe3 * H2O2 - k3 * HO * Fe2 - k4 * HO2 * Fe2 + k5 * HO2 * Fe3 - k6 * O2_minus * Fe2 + k7 * O2_minus * Fe3 - k8 * Fe2 * O2 + k9 * O2_minus * Fe3 - k10 * Fe2 * SO4_2 - k11 * SO4_minus * Fe2 + k12 * FeSO4 + k13 * FeOH2_plus * H2O2 + k14 * FeOH2_plus2 + k15 * FeOH_OH2_plus
    dFe3_dt = k1 * Fe2 * H2O2 - k2 * Fe3 * H2O2 + k3 * HO * Fe2 + k4 * HO2 * Fe2 - k5 * HO2 * Fe3 + k6 * O2_minus * Fe2 - k7 * O2_minus * Fe3 + k8 * Fe2 * O2 - k9 * O2_minus * Fe3 + k11 * SO4_minus * Fe2 - k16 * Fe3 * SO4_2 - k17 * Fe3 * SO4_2**2 + k18 * FeSO4_plus + k19 * FeSO4_2_minus - k20 * Fe3 - k21 * Fe3 - k22 * Fe3**2 + k23 * FeOH2_plus * H_plus + k24 * FeOH2_plus2 * H_plus**2 + k25 * FeOH2_plus2 * H_plus**2 - k26 * Fe3 * H2O2 + k27 * FeOH2_plus2 * H_plus
    dH2O2_dt = -k1 * Fe2 * H2O2 - k2 * Fe3 * H2O2 - k28 * H2O2 * HO + k4 * HO2 * Fe2 + k6 * O2_minus * Fe2 + k29 * HO**2 + k30 * HO2**2 + k31 * HO2 * O2_minus - k32 * HO2 * H2O2 - k33 * O2_minus * H2O2 - k34 * SO4_minus * H2O2 -k13 * FeOH2_plus * H2O2 - k26 * Fe3 * H2O2 + k27 * FeOH2_plus2 * H_plus - k35 * FeOH2_plus * H2O2 + k36 * FeOH_OH2_plus * H_plus
    dHO_dt = k1 * Fe2 * H2O2 - k28 * H2O2 * HO - k3 * Fe2 * HO - k29 * HO**2 - k37 * HO * HO2 - k38 * HO * O2_minus + k32 * HO2 * H2O2 + k33 * O2_minus * H2O2 - k39 * SO4_2 * HO - k40 * HSO4 * HO + k45 * SO4_minus + k46 * SO4_minus * OH_minus
    dHO2_dt = k2 * Fe3 * H2O2 + k28 * H2O2 * HO - k47 * HO2 + k48 * O2_minus * H_plus - k4 * HO2 * Fe2 - k5 * HO2 * Fe3 - k30 * HO2**2 + k49 * O2_minus * H_plus - k37 * HO * HO2 - k31 * HO2 * O2_minus - k32 * HO2 * H2O2 + k34 * SO4_minus * H2O2 - k50 * SO4_minus * HO2 + k13 * FeOH2_plus * H2O2 + k14 * FeHO2_2_plus + k15 * FeOH_OH2_plus
    dO2_minus_dt = k47 * HO2 - k48 * O2_minus * H_plus - k6 * O2_minus * Fe2 - k7 * O2_minus * Fe3 - k49 * O2_minus * H_plus - k38 * HO * O2_minus - k31 * HO2 * O2_minus - k33 * O2_minus * H2O2 + k8 * Fe2 * O2_minus - k9 * Fe3 * O2_minus
    dSO4_2_dt = -k10 * Fe2 * SO4_2 - k16 * Fe3 * SO4_2 - k17 * Fe3 * SO4_2**2 - k51 * H_plus * SO4_2 - k39 * SO4_2 * HO + k45 * SO4_minus + k46 * SO4_minus * OH_minus + k34 * SO4_minus * H2O2 + k50 * SO4_minus * HO2 + k11 * SO4_minus * Fe2 + k12 * FeSO4 + k18 * FeSO4_plus + k19 * FeSO4_2_minus + k52 * HSO4
    dHSO4_dt = k51 * H_plus * SO4_2 - k40 * HSO4 * HO - k52 * HSO4  
    dO2_dt = k5 * HO2 * Fe3 + k7 * O2_minus * Fe3 + k30 * HO2**2 + k37 * HO * HO2 + k38 * HO * O2_minus + k31 * HO2 * O2_minus + k32 * HO2 * H2O2 + k33 * O2_minus * H2O2 - k8 * Fe2 * O2 + k9 * Fe3 * O2_minus + k50 * SO4_minus * HO2
    dSO4_minus_dt = k39 * SO4_2 * HO + k40 * HSO4 * HO - k45 * SO4_minus - k46 * SO4_minus * HO - k34 * SO4_minus * H2O2 - k50 * SO4_minus * HO2 - k11 * SO4_minus * Fe2
    dFeSO4_dt = k10 * Fe2 * SO4_2 - k12 * FeSO4
    dFeSO4_plus_dt = k16 * Fe3 * SO4_2 - k18 * FeSO4_plus
    dFeSO4_2_minus_dt = k17 * Fe3 * SO4_2**2 - k19 * FeSO4_2_minus
    dFeOH2_plus_dt = k20 * Fe3 - k13 * FeOH2_plus * H2O2 - k23 * FeOH2_plus * H_plus + k36 * FeOH_OH2_plus * H_plus
    dFeOH2_plus2_dt = k21 * Fe3 - k24 * FeOH2_plus2 * (H_plus ** 2)
    dFeHO2_2_plus_dt = -k27 * FeHO2_2_plus * H_plus - k14 * FeHO2_2_plus
    dFe2OH2_4_plus_dt = k22 * Fe3**2 - k25 * Fe2OH2_4_plus * H_plus**2
    dFeOH_OH2_plus_dt = k35 * FeOH2_plus * H2O2 - k36 * FeOH_OH2_plus * H_plus - k15 * FeOH_OH2_plus
    dH_plus_dt = k2 * Fe3 * H2O2 + k47 * HO2 - k48 * O2_minus * H_plus + k5 * HO2 * Fe3 - k49 * O2_minus * H_plus + k34 * SO4_minus * H2O2 + k50 * SO4_minus * HO2 - k51 * H_plus * SO4_2 + k45 * SO4_minus + k20 * Fe3 + k21 * Fe3 + k22 * Fe3**2 + k13 * FeOH2_plus * H2O2 - k23 * FeOH2_plus * H_plus - k24 * FeHO2_2_plus * H_plus**2 - k25 * Fe2OH2_4_plus * H_plus**2 + k26 * Fe3 * H2O2 - k27 * FeHO2_2_plus * H_plus - k35 * FeOH2_plus * H2O2 - k36 * FeOH_OH2_plus * H_plus + k52 * HSO4
    dOH_minus_dt = k1 * Fe2 * H2O2 + k3 * Fe2 * HO + k4 * HO2 * Fe2 + k6 * O2_minus * Fe2 + k38 * HO * O2_minus + k31 * HO2 * O2_minus + k33 * O2_minus * H2O2 + k39 * SO4_2 * HO - k46 * SO4_minus * OH_minus + k13 * FeOH2_plus * H2O2 + k15 * FeOH_OH2_plus 
    Fe2 = Fe2_in + dFe2_dt 
    Fe3 = Fe3_in + dFe3_dt 
    H2O2 = H2O2_in +  dH2O2_dt 
    HO = HO_in + dHO_dt 
    HO2 = HO2_in + dHO2_dt 
    O2_minus = O2_minus_in + dO2_minus_dt 
    SO4_2 = SO4_2_in + dSO4_2_dt 
    HSO4 = HSO4_in + dHSO4_dt 
    O2 = O2_in + dO2_dt 
    SO4_minus = SO4_minus_in + dSO4_minus_dt 
    FeSO4 = FeSO4_in + dFeSO4_dt * t
    FeSO4_plus = FeSO4_plus_in + dFeSO4_plus_dt 
    FeSO4_2_minus = FeSO4_2_minus_in + dFeSO4_2_minus_dt 
    FeOH2_plus = FeOH2_plus_in + dFeOH2_plus_dt 
    FeOH2_plus2 = FeOH2_plus2_in + dFeOH2_plus2_dt 
    FeHO2_2_plus = FeHO2_2_plus_in + dFeHO2_2_plus_dt 
    Fe2OH2_4_plus = Fe2OH2_4_plus_in + dFe2OH2_4_plus_dt 
    H_plus = H_plus_in + dH_plus_dt * t
    OH_minus = OH_minus_in + dOH_minus_dt
    return [dFe2_dt, dFe3_dt, dH2O2_dt, dHO_dt, dHO2_dt, dO2_minus_dt, dSO4_2_dt, dHSO4_dt, dO2_dt, dSO4_minus_dt, dFeSO4_dt, dFeSO4_plus_dt, dFeSO4_2_minus_dt, dFeOH2_plus_dt, dFeOH2_plus2_dt, dFeHO2_2_plus_dt, dFe2OH2_4_plus_dt, dFeOH_OH2_plus_dt, dH_plus_dt, dOH_minus_dt, Fe2, Fe3, H2O2, HO, HO2, O2_minus, SO4_2, HSO4, O2, SO4_minus, FeSO4, FeSO4_plus, FeSO4_2_minus, FeOH2_plus, FeOH2_plus2, FeHO2_2_plus, Fe2OH2_4_plus, FeOH_OH2_plus, H_plus, OH_minus]



class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        
        
        self.t_fin_label = ttk.Label(self, text="Time (min):")
        self.t_fin_label.grid(column=0, row=0)
        self.t_fin_entry = ttk.Entry(self)
        self.t_fin_entry.grid(column=1, row=0)

        self.ic_label = ttk.Label(self, text="Current (A/m2):")
        self.ic_label.grid(column=0, row=1)
        self.ic_entry = ttk.Entry(self)
        self.ic_entry.grid(column=1, row=1)
        
        self.V_label = ttk.Label(self, text="Volume (m3):")
        self.V_label.grid(column=0, row=2)
        self.V_entry = ttk.Entry(self)
        self.V_entry.grid(column=1, row=2)

        self.S_label = ttk.Label(self, text="Surface (m2):")
        self.S_label.grid(column=0, row=3)
        self.S_entry = ttk.Entry(self)
        self.S_entry.grid(column=1, row=3)
        
        self.pH_in_label = ttk.Label(self, text="pH:")
        self.pH_in_label.grid(column=0, row=4)
        self.pH_in_entry = ttk.Entry(self)
        self.pH_in_entry.grid(column=1, row=4)

        self.HSO4_in_label = ttk.Label(self, text="H2SO4 (mol/m3):")
        self.HSO4_in_label.grid(column=0, row=5)
        self.HSO4_in_entry = ttk.Entry(self)
        self.HSO4_in_entry.grid(column=1, row=5)
        
        self.O2_in_label = ttk.Label(self, text="O2 (mol/m3):")
        self.O2_in_label.grid(column=0, row=6)
        self.O2_in_entry = ttk.Entry(self)
        self.O2_in_entry.grid(column=1, row=6)

        self.H2O2_in_label = ttk.Label(self, text="H2O2 (mol/m3):")
        self.H2O2_in_label.grid(column=0, row=7)
        self.H2O2_in_entry = ttk.Entry(self)
        self.H2O2_in_entry.grid(column=1, row=7)
        
        self.VM_in_label = ttk.Label(self, text="VM (mol/m3):")
        self.VM_in_label.grid(column=0, row=7)
        self.VM_in_entry = ttk.Entry(self)
        self.VM_in_entry.grid(column=1, row=7)

        self.update_button = tk.Button(self, text="Mettre à jour", command=self.on_update_button_clicked)
        self.update_button.grid(row=2, column=0, columnspan=2)
        
        self.export_button = tk.Button(self, text="Exporter en Excel", command=self.export_to_excel)
        self.export_button.grid(row=2, column=1, columnspan=2)

        self.fig, self.ax = plt.subplots()
        self.line1, = self.ax.plot([], [], label="dFe2_dt")
        self.line2, = self.ax.plot([], [], label="dH2O2_dt")
        self.ax.set_xlabel("Temps (t)")
        self.ax.set_ylabel("Solutions dFe2_dt et dH2O2_dt")
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas.get_tk_widget().grid(row=3, column=0, columnspan=2)
        self.update_graph([1, 1])
        

    def export_to_excel(self):
        if not hasattr(self, 'solution'):
            tk.messagebox.showerror("Erreur", "Aucune donnée à exporter. Cliquez sur 'Mettre à jour' pour générer les courbes.")
            return

        data = {
            "Time": self.solution.t,
            "dFe2_dt": self.solution.y[0],
            "dH2O2_dt": self.solution.y[1]
        }
        df = pd.DataFrame(data)
        df.to_excel("output_data.xlsx", index=False, engine="openpyxl")

        tk.messagebox.showinfo("Succès", "Les données ont été exportées avec succès dans output_data.xlsx")


    def on_update_button_clicked(self):
        
        
        ic = float(self.ic_entry.get())
        V = float(self.V_entry.get())
        S = float(self.S_entry.get())
        pH_in = float(self.pH_in_entry.get())
        HSO4_in = float(self.HSO4_in.get())
        O2_in = float(self.O2_in.get())
        H2O2_in = float(self.H2O2_in.get())
        self.update_graph([ic, V, S, pH_in, HSO4_in, O2_in, H2O2_in])
        

    def update_graph(self, initial_conditions):
        t_span = (0, 10)
        solution = solve_ivp(system_of_equations, t_span, initial_conditions, t_eval=np.linspace(0, 10, 100))

        self.line1.set_data(solution.t, solution.y[0])
        self.line2.set_data(solution.t, solution.y[1])
        self.line1.set_data(solution.t, solution.y[2])
        self.line2.set_data(solution.t, solution.y[3])
        self.line1.set_data(solution.t, solution.y[4])
        self.line2.set_data(solution.t, solution.y[5])
        self.line1.set_data(solution.t, solution.y[6])
        self.line2.set_data(solution.t, solution.y[7])
        self.line1.set_data(solution.t, solution.y[8])
        self.line2.set_data(solution.t, solution.y[9])
        self.line1.set_data(solution.t, solution.y[10])
        self.line2.set_data(solution.t, solution.y[11])
        self.line1.set_data(solution.t, solution.y[12])
        self.line2.set_data(solution.t, solution.y[13])
        self.line1.set_data(solution.t, solution.y[14])
        self.line2.set_data(solution.t, solution.y[15])
        self.line1.set_data(solution.t, solution.y[16])
        self.line2.set_data(solution.t, solution.y[17])
        self.line1.set_data(solution.t, solution.y[18])
        self.line2.set_data(solution.t, solution.y[19])
        self.line1.set_data(solution.t, solution.y[20])
        self.line2.set_data(solution.t, solution.y[21])
        self.line1.set_data(solution.t, solution.y[22])
        self.line2.set_data(solution.t, solution.y[23])
        self.line1.set_data(solution.t, solution.y[24])
        self.line2.set_data(solution.t, solution.y[25])
        self.line1.set_data(solution.t, solution.y[26])
        self.line2.set_data(solution.t, solution.y[27])
        self.line1.set_data(solution.t, solution.y[28])
        self.line2.set_data(solution.t, solution.y[29])
        self.line1.set_data(solution.t, solution.y[30])
        self.line2.set_data(solution.t, solution.y[31])
        self.line1.set_data(solution.t, solution.y[32])
        self.line2.set_data(solution.t, solution.y[33])
        self.line1.set_data(solution.t, solution.y[34])
        self.line2.set_data(solution.t, solution.y[35])
        self.line1.set_data(solution.t, solution.y[36])
        self.line2.set_data(solution.t, solution.y[37])
        self.line1.set_data(solution.t, solution.y[38])
        self.line2.set_data(solution.t, solution.y[39])
        self.line1.set_data(solution.t, solution.y[40])
        self.line2.set_data(solution.t, solution.y[41])
        

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

        self.solution = solution  # Stocker la solution pour l'exportation

root = tk.Tk()
app = Application(master=root)
app.mainloop()