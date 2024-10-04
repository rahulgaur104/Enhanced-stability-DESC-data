#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import FormatStrFormatter

#rho_idx = int(0)
rho_arr = np.arange(4)
omni_options = ["OP", "OT", "OH"]

for rho_idx in rho_arr:
    for keyword in omni_options:
        if keyword == "OH":
            data = np.load(
                    os.path.dirname(os.getcwd()) + f"/analysis/s-alpha-ball/ball_scan_data_OH_ball5_001_optimized_surf{rho_idx}.npy")  
            shat0 = np.array([0.05426069, 0.18837314, 0.38969486, 0.68124174])
            dP_ds0 = -1 * np.array([-0.03015929, -0.08042477, -0.13069025, -0.18095574])
    
            # Define the grid dimensions (adjust according to data)
            x = np.linspace(-0.0, 0.8, data.shape[1])
            y = np.linspace(-1, 3, data.shape[0])
            X, Y = np.meshgrid(x, y)
    
            # Create the contour plot
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X, Y, data, 30, cmap='hot')  
            plt.contour(X, Y, data, levels=[0.0000], colors='white',linewidths=2)
            cbar = plt.colorbar(contour)
        elif keyword == "OT":
            data = np.load(
                    os.path.dirname(os.getcwd()) + f"/analysis/s-alpha-ball/ball_scan_data_OT_022_optimized_surf{rho_idx}.npy")  
            shat0 = np.array([-0.03724863, -0.09633925, -0.1519759 , -0.2044528 ])
            dP_ds0 = -1 * np.array([-0.01206373, -0.0321699 , -0.0522761 , -0.0723823 ])

            # Define the grid dimensions (adjust according to data)
            x = np.linspace(-0., 0.8, data.shape[1])
            y = np.linspace(-1, 3, data.shape[0])
            X, Y = np.meshgrid(x, y)
    
            # Create the contour plot
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X, Y, data, 30, cmap='hot')
            plt.contour(X, Y, data, levels=[0.0000], colors='white',linewidths=2)
            cbar = plt.colorbar(contour)
        else:
            data = np.load(
                    os.path.dirname(os.getcwd()) + f"/analysis/s-alpha-ball/ball_scan_data_OP_033_optimized_surf{rho_idx}.npy")  
            shat0 = np.array([-0.12212944, -0.22243528, -0.22393473, -0.142355  ])
            dP_ds0 = -1 * np.array([-0.00999026, -0.02664071, -0.04329115, -0.05994159])

            # Define the grid dimensions (adjust according to data)
            x = np.linspace(-0.1, 0.3, data.shape[1])
            y = np.linspace(-1, 3, data.shape[0])
            X, Y = np.meshgrid(x, y)
    
            # Create the contour plot
            plt.figure(figsize=(8, 6))
            contour = plt.contourf(X, Y, data, 35, cmap='hot')  
            if rho_idx == 3:
                plt.contour(X, Y, data, levels=[0.001], colors='white',linewidths=2)  
            else:
                plt.contour(X, Y, data, levels=[0.0000], colors='white',linewidths=2)
            cbar = plt.colorbar(contour)
        
    
        def format_func(x, p):
                return f"{x:.4f}"  # Change 2 to the desired number of decimal places
        
        # Apply the custom formatter
        cbar.formatter = plt.FuncFormatter(format_func)
        cbar.update_ticks()
        
        cbar.ax.tick_params(labelsize=24)  # Change 14 to your desired font size
        
        plt.plot(dP_ds0[rho_idx], shat0[rho_idx], 'x', color='lime', markersize=16, markeredgewidth=4)
        plt.xlabel(r'$\alpha_{\mathrm{MHD}}$', fontsize=34)  # LaTeX for alpha
        plt.ylabel(r'$\hat{s}$', fontsize=34, labelpad=-15)  # LaTeX for hat{s}
        plt.yticks(fontsize=26)
        
        # Set x-axis ticks
        plt.xticks(np.linspace(np.min(x),np.max(x), 5), fontsize=26)
        plt.gca().set_xticklabels([f'{x:.2f}' for x in plt.gca().get_xticks()])
        plt.tight_layout()
        plt.savefig(f"ballooning_stability/{keyword}_optimized_s-alpha_rho{rho_idx}.pdf", dpi=400)
        plt.close()
    
    #plt.show()



