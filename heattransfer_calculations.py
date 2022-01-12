# Python script to calculate the heat transfer coefficients and more of the various multitubular reactor configurations.
# Authors: WB2543 Multitubular Reactor Project Group 1
# Code written by Nick van der Kroon - equations and methods by group

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def equivalent_diameter_triangular(d_o):
    # equation from the 'Kern method'
    pitch = 1.33 * d_o
    d_e = 4 * ((np.sqrt(3) * (pitch ** 2)) / 4 - np.pi * d_o ** 2 / 8) / (np.pi * d_o / 2)
    return d_e

def Reynolds_number(d_array, u_array, rho, mu):
    # Returns a nxm numpy array upon being supplied with n values for d and m values for u.
    # It is important that d_array and u_array are supplied as numpy arrays, even if they are a single value
    # Follows the standard formula for Reynolds numbers, Re = u * d * rho / mu
    Re = np.empty((d_array.size, u_array.size))
    for i in range(d_array.size):
        for j in range(u_array.size):
            Re[i,j] = d_array[i] * u_array[j] * rho / mu
    return Re

def h_cc_square_crossflow(d_range, Re_c):
    # Given Reynolds values are all below 10^4
    # Equation '[75] - in line' from 'Advances in Modelling and Design of Multitubular Fixed- Bed  Reactors' is used here
    h_cc = np.zeros_like(Re_c) # intermediate array where the heat transfer coefficient is stored

    i = 0 # the 'i' iterator indicates the row, and as a result it also indicates the d_range value we want
    for row in Re_c:
        j = 0 # the iterators are also used to store the data to the proper indices
        for value in row:
            h_cc[i, j] = 0.27 * (Re_c[i, j] ** 0.63) * (Pr_c ** 0.36) * lambda_c / d_range[i]
            # print("value: ", h_cc[i, j], "created with Re_c: ", Re_c[i, j], "and d: ", d_range[i])
            j += 1 # next flow velocity 'u'
        i += 1 # next diameter 'd'
    return h_cc

def h_cc_triangular_crossflow(d_range, Re_c):
    # Triangular crossflow follows the same solution as square crossflow, however a different equation is used
    # The Reynolds number is below 10^4 for all relevant cases, so we use equation '[75] - staggered 'triangular''
    h_cc = np.zeros_like(Re_c)

    i = 0
    for row in Re_c:
        j = 0
        for value in row:
            h_cc[i, j] = 0.35 * (2 / np.sqrt(3)) ** 0.2 * (Re_c[i, j] ** 0.6) * (Pr_c ** 0.36) * lambda_c / d_range[i]
            j += 1
        i += 1
    return h_cc

def h_L_triangular_parallel(d_eL_range, Re_eL):
    # Reynolds numbers used here are all below 10^4, so the best fitting equation is [86]
    h_L = np.zeros_like(Re_eL)

    i = 0
    for row in Re_eL:
        j = 0
        for value in row:
            # uses equation [86] here
            # Pr_c / Pr_cw is assumed to be 1
            h_L[i, j] = 0.021 * (Re_eL[i, j] ** 0.8) * (Pr_c ** 0.43) * lambda_c / d_eL_range[i]
            j += 1
        i += 1
    return h_L

def h_reacting_gas_to_wall(d_t, d_p, Re_p):
    # Reynolds numbers used here range from ~60 to ~360
    # Therefor we use equation [26], and use the first equation in the table since d_p / d_t > falls within the range for all cases
    # This function returns a nxm numpy array containing the heat transfer coefficient for each permutation of d_t and Re_p

    U = np.zeros((d_t.size, Re_p.size))
    # The Reynolds number Re_p here is a function of gas flow velocity
    # The diameter remains constant for each row, and the gas flow velocity (thus Re_p value) remains constant for each column
    i = 0
    for row in U:
        j = 0
        for value in row:
            U[i, j] = 2.09 * np.exp(-6 * d_p / d_t[i]) * Re_p[0, j] ** 0.8 * lambda_f / d_t[i]
            j += 1
        i += 1
    return U

def overall_heat_transfer_coefficient(d_range, u_range, h_internal, h_external):
    # Calculates the overall heat transfer coefficient U as given by the powerpoint 'Multitubular-Reactor-Design -Basics' slide 39
    # Where h_0 is the external heat transfer coefficient of the coolant setup, and h_1 is the internal coefficient of the reacting gas
    # h_0 varies only with diameter, but h_1 varies with both diameter and velocity
    # diameter will remain constant along the rows, velocity along the columns

    U = np.zeros((d_range.size, u_range.size))

    i = 0
    for row in U:
        j = 0
        for value in row:
            # h_external is always a mx1 array since we only have one coolant velocity
            # but h_internal is mxn since it varies with both reacting gas velocity and diameter
            U[i, j] = (1 / h_internal[i, j] + 1 / h_external[j] + (d_o[i] - d_i[i]) / (2 * lambda_w)) ** -1

            j += 1
        i += 1
    
    return U

def three_dimensional_graph_h_coeff(d_range, u_range, h_coeff, figtitle):
    # this function creates a pyplot graph
    X, Y, Z = [], [], []

    # X holds diameter data, Y holds flow velocity, Z holds the heat transfer coefficients
    i = 0
    for row in h_coeff:
        j = 0
        for value in row:
            X.append(d_range[i])
            Y.append(u_range[j])
            Z.append(h_coeff[i, j])
            j += 1
        i += 1

    # Convert to numpy arrays since they are needed by certain matplotlib functions
    X = np.array(X)
    Y = np.array(Y)
    Z = np.array(Z)

    fig = plt.figure()
    fig.suptitle(figtitle)
    ax = plt.axes(projection='3d')
    ax.scatter(X, Y, Z, c=Z, )
    #ax.plot_trisurf(X, Y, Z)
    ax.set_xlabel("tube diameter 'd' (m)")
    ax.set_ylabel("gas velocity 'u' (m/s)")
    ax.set_zlabel("heat transfer coefficient")

def export_U_values(U_sq_cross, U_tri_cross, U_tri_parallel):
    # Export the values to numpy arrays for use in numericalsolver.py
    ### IMPORTANT: The values of u_f and d must match the steps on the slider in numericalsolver.py
    Path("./data/").mkdir(parents=True, exist_ok=True) # Make directory if it doesn't yet exist
    np.save('data/U_sq_cross.npy', U_sq_cross)
    np.save('data/U_tri_cross.npy', U_tri_cross)
    np.save('data/U_tri_parallel.npy', U_tri_parallel)
    return


if __name__ == "__main__": # Script starts here
    # Here we define constant used in calculations throughout the script
    ### Tube and reactor properties
    d = np.arange(0.02, 0.044, 0.004)       # m, range of tube diameters
    w = 0.0015                              # m, tube wall thickness
    d_i = d - w/2                           # m, internal tube diameter
    d_o = d + w/2                           # m, external tube diameter
    d_eL_triangular = equivalent_diameter_triangular(d) # m, equivalent tube diameter in triangular arrangement
    lambda_w = 14.4                         # W/(m*K), thermal conductivity of tube wall. Assumed 'Steel - Stainless, Type 304' at 20 deg C
    d_p = np.array([0.003])                 # m, catalyst particle diameter
    
    #u = np.linspace(0.5, 3, 11)             # m/s, range of gas superficial velocities

    ### Coolant properties
    u_c = np.array([0.2])                   # m/s, coolant linear velocity
    rho_c = 1791                            # kg/m^3, coolant density
    mu_c = 0.0019                           # Ns/m^2, coolant viscosity
    c_pc = 1562                             # J/(kg*K), coolant specific heat
    lambda_c = 0.331                        # W/(m*K), coolant thermal conductivity
    Pr_c = mu_c * c_pc / lambda_c           # 1, coolant Prandtl number

    ### Reacting gas properties
    lambda_f = 0.0488                       # W/(m*K), reacting gas thermal conductivity
    rho_f = 1.225                           # kg/m3, density of air at ISA. Since the reactor uses a flow of air with injected benzene, we assume air density for reacting gas density.
    mu_f = 0.000031                         # Ns/m^2, reacting gas viscosity
    u_f = np.arange(0.5, 3.5, 0.5)          # m/s, range of reacting gas superficial velocities



    # Get the Reynolds numbers here
    # Since they are functions of both d and u, the Re_x variables will be numpy matrices containing the results
    # d will be constant for the rows, u for the columns
    Re_c = Reynolds_number(d, u_c, rho_c, mu_c)
    Re_eL_triangular = Reynolds_number(d_eL_triangular, u_c, rho_c, mu_c)
    Re_p = Reynolds_number(d_p, u_f, rho_f, mu_f)

    # Here we calculate the tube-to-coolant heat transfer coefficient for the various configurations we will be investigating
    h_cc_sq_crossflow = h_cc_square_crossflow(d, Re_c)
    h_cc_tri_crossflow = h_cc_triangular_crossflow(d, Re_c)
    h_L_tri_parallel = h_L_triangular_parallel(d_eL_triangular, Re_eL_triangular)
    
    h_reacting_gas = h_reacting_gas_to_wall(d_i, d_p, Re_p)
    U_sq_cross = overall_heat_transfer_coefficient(d, u_f, h_reacting_gas, h_cc_sq_crossflow)
    U_tri_cross = overall_heat_transfer_coefficient(d, u_f, h_reacting_gas, h_cc_tri_crossflow)
    U_tri_parallel = overall_heat_transfer_coefficient(d, u_f, h_reacting_gas, h_L_tri_parallel)

    ### 2D graphing of coolant heat transfer coefficients
    fig2d = plt.figure()
    fig2d.suptitle('coolant heat transfer coefficients')
    axes2d = plt.axes()
    axes2d.plot(d, h_cc_sq_crossflow)
    axes2d.plot(d, h_cc_tri_crossflow)
    axes2d.plot(d, h_L_tri_parallel)
    axes2d.legend(["Square crossflow", "Triangular crossflow", "Triangular parallel flow"])
    axes2d.set_xlabel("tube diameter 'd' (m)")
    axes2d.set_ylabel("heat transfer coefficient 'h_c'")

    ### 3D graphing of overall heat transfer coefficients
    # The 3D graphs are best viewed as pop-outs instead of within the python command line terminal.
    three_dimensional_graph_h_coeff(d, u_f, h_reacting_gas, 'reacting gas heat transfer coefficient to wall')
    three_dimensional_graph_h_coeff(d, u_f, U_sq_cross, 'overall heat transfer coefficient square crossflow')
    three_dimensional_graph_h_coeff(d, u_f, U_tri_cross, 'overall heat transfer coefficient triangular crossflow')
    three_dimensional_graph_h_coeff(d, u_f, U_tri_parallel, 'overall heat transfer coefficient triangular parallel flow')

    export_U_values(U_sq_cross, U_tri_cross, U_tri_parallel)

    #print('tricross: at d:', d[5], " u:", u_f[8], " Re_c:", Re_c[5], " Re_p:", Re_p[0, 8], " h_cc:", h_cc_tri_crossflow[5], " h_r:", h_reacting_gas[5, 8], " and U:", U_tri_cross[5, 8])
    # Above print statement for data validation: manually run the numbers to make sure the script returns the correct data!
    plt.show()


    print('debug halt')
