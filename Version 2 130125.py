import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import statistics as stats  ##This may become irrelavant with fitting
import scipy.optimize as optimize


### Variables ###
Type = 'B002'
Ionisation = 'Ne 1e13 20keV'
Perovskite = 'FA0.87Cs0.13PbI3'
Substrate = 'SiO2'
Version = ''
LifeTime_Version = 3
Test = 'Slow'
Colour = 'White'
Chosen_Data_Number_Before = 1
Chosen_Data_Number_After = 1
Max_Data_Number_Before = 4
Max_Data_Number_After = 4
Start_Angle = 11
End_Angle = 12.2
Start_Wavelength = 420
End_Wavelength = 900
Start_Time = 190
End_Time = 300
Predicted_LifeTime = 10
Predicted_Beta = 1
Normalise = False
Select_All = False
Normalised_Gaussian = True
Zero_Time = True
Normalised_Exponential = False
Look_At_One_XRD = True
Compare_Peaks = True
Normal_plots_only = False
PLQE_Wavelength = '405 nm'
PLQE_Power = '2.2 mW'
WavelengthEm_Before = '370 nm'
WavelengthEm_After  = '370 nm'
WavelengthEx_Before = '780 nm'
WavelengthEx_After  = '790 nm'
Em = 'Em'
WavelengthEx = '505 nm'
Ex = 'Ex'
High_Power_Density_Before = 'High Power Density'
High_Power_Density_After =  'High Power Density'

Ionisation_XRD = Ionisation.replace('20 keV', '20keV')
Ionisation_XRD = Ionisation_XRD.replace(' ','_')
#### General definitions ####
def Normalised_Normalised_Gaussian_XRD(Angle, Center, Stdev):
    a = np.exp((-(Angle-Center)**2)/(2*Stdev**2))
    return a

def Ratio_Gaussian_XRD_Equation(Angle, Center, Stdev, Height, Min):
    return (Height-Min)*np.exp((-((Angle-Center)**2))/(2*Stdev**2)) + Min

def Normalised_Exponential_Equation(t, Tau, y0, Beta):
    """Normalized exponential equation."""
    return np.exp(-(t / Tau)**Beta) + y0


def Normalised_Gaussian_XRD(df, Before_After):
    x_value = df.iloc[:,0]
    y_value = df.iloc[:,1]
    if Normalised_Gaussian == True:
        Max_Index = np.argmax(y_value)
        Predicted_Centre = x_value[Max_Index]
        Standard_Deviation = stats.stdev(y_value)
    
        Guess = [Predicted_Centre,Standard_Deviation]
        poptb, _ = optimize.curve_fit(Normalised_Normalised_Gaussian_XRD, x_value, y_value, p0=Guess)
        Fitted_Centre, Fitted_Stdev = poptb
        Fitted_Normalised_Gaussian = Normalised_Normalised_Gaussian_XRD(x_value,Fitted_Centre,Fitted_Stdev)
        if Before_After == 'Before Ion':
            plt.plot(x_value, Fitted_Normalised_Gaussian, label='Before; Fit',color = 'lime')
        else:
            plt.plot(x_value, Fitted_Normalised_Gaussian, label='After; Fit',color = 'Black')
            
def Ratio_Gaussian_XRD(df, Before_After, Max, Colorful):
    x_value = df.iloc[:,0]
    y_value = df.iloc[:,1]
    if Normalised_Gaussian == True:
        Predicted_centre = 0
        Standard_Deviation = 0.2
        Min = np.min(df.iloc[:,1])
        Guess = [Predicted_centre, Standard_Deviation, Max, Min]
        poptb, _ = optimize.curve_fit(Ratio_Gaussian_XRD_Equation, x_value, y_value, p0 = Guess)
        Fitted_Centre, Fitted_Stdev, Fitted_Height, Fitted_Min = poptb
        Fitted_Gaussian = Ratio_Gaussian_XRD_Equation(x_value, Fitted_Centre, Fitted_Stdev, Fitted_Height, Fitted_Min)
        plt.plot(x_value, Fitted_Gaussian, label = '{}'.format(Before_After), color = Colorful, linestyle='--')
        print('---------')
        print('{}'.format(Before_After))
        print('Predicted Centre {}, Fitted Centre {}'.format(Predicted_centre,Fitted_Centre))
        print('Predicted Stdev {}, Fitted Stdev {}'.format(Standard_Deviation,Fitted_Stdev))
        print('Predicted Height {}, Fitted Height {}'.format(Max, Fitted_Height))
        print('Predicted Min {}, Fitted Min {}'.format(Min,Fitted_Min))

        

   
def Normalised_Exponential_fit(df, Before_After, Predicted_LifeTime, Predicted_Beta, y):
    """Fits normalized exponential to the data in df."""
    y0 = 0.01
    x_value = df.iloc[:, 0]
    y_value = df.iloc[:, 1]
    
    if Normalised_Exponential == True:
        # Initial guess for parameters [LifeTime, y0, Beta]
        Guess = [Predicted_LifeTime, y0, Predicted_Beta]
        
        # Curve fitting with adjusted bounds for all parameters
        popt, pcov = optimize.curve_fit(
            Normalised_Exponential_Equation, x_value, y_value, p0=Guess, 
            maxfev=2000000, bounds=([0, 0, 0], [np.inf, 1, np.inf])  # Adjust bounds for Beta as well
        )
        
        Fitted_LifeTime, Fitted_y0, Fitted_Beta = popt
        Fitted_Normalised_Exponential = Normalised_Exponential_Equation(x_value, Fitted_LifeTime, Fitted_y0, Fitted_Beta)
        
        # Calculate standard errors
        perr = np.sqrt(np.diag(pcov))
        Error_LifeTime, Error_y0, Error_Beta = perr
        
        # Plotting the fitted curve
        label = f'{Before_After}; Fit'
        color = 'Lime' if Before_After == 'Before Ion' else 'black'
        plt.plot(x_value, Fitted_Normalised_Exponential, label=label, color=color)
        
        # Print the fitted Lifetime and errors
        print('{} LifeTime: {:.3f} ± {:.3f}'.format(Before_After, Fitted_LifeTime, Error_LifeTime))
        print('{} Beta: {:.3f} ± {:.3f}'.format(Before_After, Fitted_Beta, Error_Beta))
        
        plt.text(0.1, y, f'{Before_After} LifeTime: {Fitted_LifeTime:.2f} ± {Error_LifeTime:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12, color=color)
        plt.text(0.1, y-0.05, f'{Before_After} Beta: {Fitted_Beta:.2f} ± {Error_Beta:.2f}', 
                 transform=plt.gca().transAxes, fontsize=12, color=color)  
        
def absorbance(direct, indirect):
    all(direct.iloc[:,0] == indirect.iloc[:,0])  # check that is never used for the same x-axis
    absorbance = 2 - np.log10((direct.iloc[:,1] / indirect.iloc[:,1]) * 100)
    return absorbance

def select_x_range2(dataframe, lower_boundary, upper_boundary):
    # Select x_range
    df = dataframe
    condition = (df > lower_boundary) & (df < upper_boundary)
    result = df[condition]
    return result

def select_x_range(dataframe, lower_boundary, upper_boundary):
    # Select x_range
    df = dataframe
    condition = (df[df.columns[0]] > lower_boundary) & (df[df.columns[0]] < upper_boundary)
    result = df[condition]
    return result

def create_array(up_to):   ##Given a number, make an array up to it  
    """
    Creates a list of integers from 0 up to the specified input (exclusive).
    """
    if up_to < 0:
        raise ValueError("Input must be a non-negative integer.")
    return list(range(up_to+1))

### File Finders ###

def File_XRD(Type, Ionisation_XRD, Perovskite, BeforeAfter, Substrate, Test, Version, DataNumber):
    """Returns filename for wanted test
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne_1e13_20keV', 'Ar_1e13_20keV.
        Perovskite: 'FA0.87Cs0.13PbI3'.
        BeforeAfter: 'Before Ion', 'After Ion'.
        Substrate: 'SiO2'.
        Test: 'Rocking', 'Slow'
        Version: '','2','3'
        DataNumber: '0','1','2','3','4'.
    """
    # List of components, including an empty string
    components = [str(Type), str(Ionisation_XRD), str(Perovskite), str(BeforeAfter), str(Substrate), str(Test), str(Version)]
    
    # Filter out any empty components and join the remaining ones with underscores
    combined_string = "XRD Data/"+"_".join(component for component in components if component)
    combined_string = combined_string + "/" + 'Data'+ str(DataNumber)
    combined_string = combined_string + "/" + 'Profile' + str(DataNumber)
    return combined_string

def File_Absorbance(Type, Ionisation, BeforeAfter, Directness, Version, Colour):
    """
    Args:.
        Type: 'BXXX','Blank'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        BeforeAfter: 'Before Ion', 'After Ion'.
        Directness: 'Direct', 'Indirect','Direct1','Indirect1','Direct2', and so on.
        Version: '', '2', '3'.
        Color: 'White'.
    """
    # List of components, including an empty string
    components = [str(Type), str(Ionisation), str(Directness+Version), str(Colour)]
    
    # Filter out any empty components and join the remaining ones with underscores
    combined_string = 'Absorbance ' + BeforeAfter + "/" + " ".join(component for component in components if component)
    combined_string = combined_string +'.csv'
    return combined_string

def File_LifeTime(Type, Ionisation, BeforeAfter, WavelengthEm, WavelengthEx, High_Power_Density, LifeTime_Version):
    """Returns filename for wanted test, CAP SENSITIVE
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        BeforeAfter: 'Before Ion', 'After Ion'.
        WavelengthEm: '370 nm', '780 nm', '790 nm'.
        WavelengthEx: '505'.
        High Power density: 'High Power Density', 'HPD', ''.
        LifeTime_Version: 1, 2, 3.
    """
    
    # List of components, including an empty string

    components = [str(Type), str(Ionisation), str(WavelengthEm), 'Em', str(WavelengthEx), 'Ex', 'LFT' + str(LifeTime_Version), str(High_Power_Density)]

    
    # Filter out any empty components and join the remaining ones with underscores
    combined_string = 'Em Ex and Lifetimes ' + BeforeAfter + "/" + " ".join(component for component in components if component)
    combined_string = combined_string +'.txt'
    return combined_string

def File_Em_Ex(Type, Ionisation, BeforeAfter, Wavelength, Em_Ex):
    """Returns filename for wanted test, CAP SENSITIVE
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        BeforeAfter: 'Before Ion', 'After Ion'.
        Wavelength: '370 nm', '780 nm', '790 nm'.
        Em_Ex: 'Em', 'Ex'
    """
    
    # List of components, including an empty string

    components = [str(Type), str(Ionisation), str(Wavelength), str(Em_Ex)]

    
    # Filter out any empty components and join the remaining ones with underscores
    combined_string = 'Em Ex and Lifetimes ' + BeforeAfter + "/" + " ".join(component for component in components if component)
    combined_string = combined_string +'.txt'
    return combined_string


def File_PLQE(Type, Ionisation, BeforeAfter, Directness, Version, Wavelength, Power):
    """Returns filename for wanted test, CAP SENSITIVE
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        Directness: 'Blank', 'Direct', 'Indirect'.
        Version: '','2','3'.
        BeforeAfter: 'Before Ion', 'After Ion'.
        WavelengthEm: '405 nm'
        Power: '2.2 mW'
    """
    # List of components, including an empty string
    components = [str(Type), str(Ionisation), str(Directness+Version), str(Wavelength), str(Power)]
    
    # Filter out any empty components and join the remaining ones with underscores
    combined_string = 'PLQE ' + BeforeAfter + "/" + " ".join(component for component in components if component)
    combined_string = combined_string +'.csv'
    return combined_string

### Data Frame Makers
def df_Maker_XRD(Type, Ionisation_XRD, Perovskite, BeforeAfter, Substrate, Test, Version, DataNumber):
    """
    .txt to dataframe

    Args:
        Type: 'BXXX'.
        Ionisation_XRD: 'Empty', 'Control', 'Ne_1e13_20keV', 'Ar_1e13_20keV.
        Perovskite: 'FA0.87Cs0.13PbI3'.
        Substrate: 'SiO2'.
        Test: 'Rocking', 'Slow'.
        Version: '', '2', '3'.
        DataNumber: '0','1','2','3','4'.
    """
    df = pd.read_csv(File_XRD(Type, Ionisation_XRD, Perovskite, BeforeAfter, Substrate, Test, Version, DataNumber)+'.txt', sep='\t')
    return df



def df_Maker_Absorbance(Type, Ionisation, BeforeAfter, Directness, Version, Colour):
    """
    .txt to dataframe
    Args:
        Type: 'BXXX','Blank'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20keV', 'Ar 1e13 20keV, '',
        BeforeAfter: 'Before Ion', 'After Ion'.
        Directness: 'Direct', 'Indirect', 'Direct1' ,'Indirect1' ,'Direct2' , and so on.
        Color: 'White'.
    """
    File_Name = File_Absorbance(Type, Ionisation, BeforeAfter, Directness, Version, Colour)
    df = pd.read_csv(File_Name)
    return df

def df_Maker_LifeTime(Type, Ionisation, BeforeAfter, WavelengthEmxx, WavelengthExxx, High_Power_Density, LifeTime_Version):
    """
    .txt to dataframe
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        BeforeAfter: 'Before Ion', 'After Ion'.
        WavelengthEm: '370 nm', '780 nm', '790 nm'.
        WavelengthEx: '505 nm'.
        High Power density: 'High Power Density', 'HPD', ''.
        LifeTime_Version: 1, 2, 3.
    """
    File_Name = File_LifeTime(Type, Ionisation, BeforeAfter, WavelengthEmxx, WavelengthExxx, LifeTime_Version, High_Power_Density)
    df = pd.read_csv(File_Name, sep=',')
    return df

def df_Maker_Em_Ex(Type, Ionisation, BeforeAfter, Wavelength, Em_Ex):
    """
    .txt to dataframe
    Args:
        Type: 'BXXX'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20 keV', 'Ar 1e13 20keV, ''.
        BeforeAfter: 'Before Ion', 'After Ion'.
        Wavelength_Before: '370 nm', '780 nm', '790 nm'.
        Wavelength_After:  '370 nm', '780 nm', '790 nm'.
        Em_Ex: 'Em', 'Ex'.
    """
    File_Name = File_Em_Ex(Type, Ionisation, BeforeAfter, Wavelength, Em_Ex)
    df = pd.read_csv(File_Name, sep=',')
    return df

def df_Maker_PLQE(Type, Ionisation, Directness, Version, BeforeAfter, Wavelength, Power):
    """
    .txt to dataframe
    Args:
        Type: 'BXXX','Blank'.
        Ionisation: 'Empty', 'Control', 'Ne 1e13 20keV', 'Ar 1e13 20keV, '',
        BeforeAfter: 'Before Ion', 'After Ion'.
        Directness: 'Blank', 'Direct', 'Indirect'.
        Version: '', '2', '3'.
        Power: '2.2 mW'.
    """
    File_Name = File_PLQE(Type, Ionisation, BeforeAfter, Directness, Version, Wavelength, Power)
    df = pd.read_csv(File_Name)
    return df


### Graphers ###
def Compare_Before_After_XRD(
            Type, Ionisation_XRD, Perovskite, Substrate, Version, Test,
            Chosen_Data_Number_Before, Chosen_Data_Number_After, Start_Angle, End_Angle,
            Normalise, Select_All, Normalised_Gaussian, Look_At_One_XRD, Compare_Peaks, Normal_plots_only):

    Data_Numbers_Before = []
    Data_Numbers_After  = []
    if Compare_Peaks == True:
        
        Before_df_1 = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, 1)
        After_df_1  = df_Maker_XRD(Type, Ionisation_XRD, Perovskite, 'After Ion', Substrate, Test, Version, 1)
        Before_df_2 = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, 2)
        After_df_2  = df_Maker_XRD(Type, Ionisation_XRD, Perovskite, 'After Ion', Substrate, Test, Version, 2)
        
        


        Before_df = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, 1)
        After_df = df_Maker_XRD(Type, Ionisation_XRD, Perovskite, 'After Ion', Substrate, Test, Version, 1)
        if np.isclose(After_df.iloc[:, 0], 12.75).any():
            After_df_1 = select_x_range(After_df, 11-0.01, 12.5+0.01)
            After_df_2 = select_x_range(After_df, 13-0.01, 15+0.01)
            After_df_1 = After_df_1.reset_index(drop=True)
            After_df_2 = After_df_2.reset_index(drop=True)
            
            """
        if Condition1:
            print('conl1')
            Before_df_1 = select_x_range(Before_df, 11, 12.5)
            After_df_1 = select_x_range(After_df, 11, 12.5)
            
        if Condition2:
            print('conl2')
            print(Before_df_2)
            print(After_df_2)
            Before_df_2 = select_x_range(Before_df, 13, 15)
            After_df_2 = select_x_range(After_df, 13, 15)"""




        Before_Range_Angle_1 = np.max(Before_df_1.iloc[:,0]) - np.min(Before_df_1.iloc[:,0])
        After_Range_Angle_1  = np.max(After_df_1.iloc[:,0])  - np.min(After_df_1.iloc[:,0])
        Before_Range_Angle_2 = np.max(Before_df_2.iloc[:,0]) - np.min(Before_df_2.iloc[:,0])
        After_Range_Angle_2  = np.max(After_df_2.iloc[:,0])  - np.min(After_df_2.iloc[:,0])
        
        Before_Mid_Angle_1 = (np.max(Before_df_1.iloc[:,0]) + np.min(Before_df_1.iloc[:,0])) /2
        After_Mid_Angle_1  = (np.max(After_df_1.iloc[:,0])  + np.min(After_df_1.iloc[:,0]))  /2
        Before_Mid_Angle_2 = (np.max(Before_df_2.iloc[:,0]) + np.min(Before_df_2.iloc[:,0])) /2
        After_Mid_Angle_2  = (np.max(After_df_2.iloc[:,0])  + np.min(After_df_2.iloc[:,0]))  /2
        
        Num = 1-1/np.exp (1)
        Angle_Boundary_Lower_Before_1, Angle_Boundary_Upper_Before_1 = Before_Mid_Angle_1 - Num*Before_Range_Angle_1/2, Before_Mid_Angle_1 + Num*Before_Range_Angle_1/2
        Angle_Boundary_Lower_After_1,  Angle_Boundary_Upper_After_1  = After_Mid_Angle_1  - Num*After_Range_Angle_1/2,  After_Mid_Angle_1  + Num*After_Range_Angle_1 /2
        Angle_Boundary_Lower_Before_2, Angle_Boundary_Upper_Before_2 = Before_Mid_Angle_2 - Num*Before_Range_Angle_2/2, Before_Mid_Angle_2 + Num*Before_Range_Angle_2/2
        Angle_Boundary_Lower_After_2,  Angle_Boundary_Upper_After_2  = After_Mid_Angle_2  - Num*After_Range_Angle_2/2,  After_Mid_Angle_2  + Num*After_Range_Angle_2 /2
        
        Before_df_1_Numbound = select_x_range(Before_df_1, Angle_Boundary_Lower_Before_1, Angle_Boundary_Upper_Before_1)       
        After_df_1_Numbound  = select_x_range(After_df_1 , Angle_Boundary_Lower_After_1 , Angle_Boundary_Upper_After_1 )
        Before_df_2_Numbound = select_x_range(Before_df_2, Angle_Boundary_Lower_Before_2, Angle_Boundary_Upper_Before_2)       
        After_df_2_Numbound  = select_x_range(After_df_2 , Angle_Boundary_Lower_After_2 , Angle_Boundary_Upper_After_2 )       
        
        Max_Bounded_Before_1 = np.max(Before_df_1_Numbound.iloc[:,1])
        Max_Bounded_After_1  = np.max(After_df_1_Numbound.iloc[:,1] )
        Max_Bounded_Before_2 = np.max(Before_df_2_Numbound.iloc[:,1])
        Max_Bounded_After_2  = np.max(After_df_2_Numbound.iloc[:,1] )
        
        
        
        if Normalise == True: 
            Intensity_Before_1 = (Before_df_1.iloc[:,1] - np.min(Before_df_1.iloc[:,1])) / (Max_Bounded_Before_1 - np.min(Before_df_1.iloc[:,1]))
            Intensity_Before_2 = (Before_df_2.iloc[:,1] - np.min(Before_df_2.iloc[:,1])) / (Max_Bounded_Before_2 - np.min(Before_df_2.iloc[:,1]))
            Intensity_After_1  = (After_df_1.iloc[:,1]  - np.min(After_df_1.iloc[:,1]) ) / (Max_Bounded_After_1  - np.min(After_df_1.iloc[:,1]) )
            Intensity_After_2  = (After_df_2.iloc[:,1]  - np.min(After_df_2.iloc[:,1]) ) / (Max_Bounded_After_2  - np.min(After_df_2.iloc[:,1]) )
            
            
        else:
            Intensity_Before_1 = Before_df_1.iloc[:,1]
            Intensity_Before_2 = Before_df_2.iloc[:,1]
            Intensity_After_1  = After_df_1.iloc[:,1]
            Intensity_After_2  = After_df_2.iloc[:,1]   
            
        Max_Intensity_Before_1 = Max_Bounded_Before_1
        Max_Intensity_After_1  = Max_Bounded_After_1
        Max_Intensity_Before_2 = Max_Bounded_Before_2
        Max_Intensity_After_2  = Max_Bounded_After_2
        
        Max_Intensity_Index_Before_1 = Before_df_1[Before_df_1.iloc[:, 1] == Max_Intensity_Before_1].index[0]
        Max_Intensity_Index_After_1  = After_df_1[After_df_1.iloc[:, 1] == Max_Intensity_After_1].index[0]
        Max_Intensity_Index_Before_2 = Before_df_2[Before_df_2.iloc[:, 1] == Max_Intensity_Before_2].index[0]
        Max_Intensity_Index_After_2  = After_df_2[After_df_2.iloc[:, 1] == Max_Intensity_After_2].index[0]
        print(Max_Intensity_Index_After_2)


        
        Max_Angle_Before_1 = Before_df_1.iloc[Max_Intensity_Index_Before_1, 0 ]
        Max_Angle_After_1  = After_df_1.iloc[Max_Intensity_Index_After_1,   0 ]
        Max_Angle_Before_2 = Before_df_2.iloc[Max_Intensity_Index_Before_2, 0 ]
        Max_Angle_After_2  = After_df_2.iloc[Max_Intensity_Index_After_2,   0 ]

        
        
        Before_df_1.iloc[:,0] = Before_df_1.iloc[:,0] - Max_Angle_Before_1
        After_df_1.iloc[:,0]  = After_df_1.iloc[:,0]  - Max_Angle_After_1
        Before_df_2.iloc[:,0] = Before_df_2.iloc[:,0] - Max_Angle_Before_2
        After_df_2.iloc[:,0]  = After_df_2.iloc[:,0]  - Max_Angle_After_2
        
        Before_df_1 = pd.DataFrame({'angle': Before_df_1.iloc[:,0], 'intensity': Intensity_Before_1}, columns=['angle', 'intensity'])
        After_df_1  = pd.DataFrame({'angle': After_df_1.iloc[:,0] , 'intensity': Intensity_After_1} , columns=['angle', 'intensity'])
        Before_df_2 = pd.DataFrame({'angle': Before_df_2.iloc[:,0], 'intensity': Intensity_Before_2}, columns=['angle', 'intensity'])
        After_df_2  = pd.DataFrame({'angle': After_df_2.iloc[:,0] , 'intensity': Intensity_After_2} , columns=['angle', 'intensity'])       
        
        plt.figure(figsize = (8, 6))
        if Normal_plots_only == False:
            plt.plot(Before_df_1.angle, Before_df_1.intensity, label='Before1',color = 'b')
            plt.plot(Before_df_2.angle, Before_df_2.intensity, label='Before2',color = 'lightblue')
            plt.plot(After_df_1.angle , After_df_1.intensity , label='After1' ,color = 'r')
            plt.plot(After_df_2.angle , After_df_2.intensity , label='After2' ,color = 'lightcoral')
        
        Ratio_Gaussian_XRD(Before_df_1, 'Before 1', Max_Intensity_Before_1, 'b')
        Ratio_Gaussian_XRD(Before_df_2, 'Before 2', Max_Intensity_Before_2, 'lightblue')
        Ratio_Gaussian_XRD(After_df_1 , 'After 1' , Max_Intensity_After_1 , 'r')
        Ratio_Gaussian_XRD(After_df_2 , 'After 2' , Max_Intensity_After_2 , 'lightcoral')
        print('------------------')
        print(Max_Intensity_Before_1)
        print(Max_Intensity_Before_2)
        print(Max_Intensity_After_1)
        print(Max_Intensity_After_2)
        print(Before_df_1)
        angle = Before_df_1.angle
        print(angle)
        plt.legend()
        plt.title(r'XRD: Ratio comparison, before and after in {} ({})'.format(Type,Ionisation))
        plt.xlabel("Angle")
        plt.ylabel("Intensity")
        plt.tight_layout()
        
        plt.show()
        
            
            
    else:
        if Look_At_One_XRD == True:
            
            plt.figure(figsize = (8, 6))
            Before_df = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, Chosen_Data_Number_Before)
            After_df = df_Maker_XRD(Type, Ionisation_XRD, Perovskite, 'After Ion', Substrate, Test, Version, Chosen_Data_Number_After)
            Condition = np.max(Before_df.iloc[:, 0]) < np.max(After_df.iloc[:, 0])
            if Condition:
                Before_df_Plus_1 = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, Chosen_Data_Number_Before+1)
            
                   
            if Normalise == True:
                Intensity_Before        = (Before_df.iloc[:,1]        - np.min(Before_df.iloc[:,1]))        / (np.max(Before_df.iloc[:,1])       -np.min(Before_df.iloc[:,1])       )
                if Condition:
                    Intensity_Before_Plus_1 = (Before_df_Plus_1.iloc[:,1] - np.min(Before_df_Plus_1.iloc[:,1])) / (np.max(Before_df_Plus_1.iloc[:,1])-np.min(Before_df_Plus_1.iloc[:,1]))
                Intensity_After         = (After_df.iloc[:,1]         - np.min(After_df.iloc[:,1]))         / (np.max(After_df.iloc[:,1])        -np.min(After_df.iloc[:,1])        )
    
    
            else:
                Intensity_After = After_df.iloc[:,1]
                Intensity_Before = Before_df.iloc[:,1]
                if Condition:
                    Intensity_Before_Plus_1 = Before_df_Plus_1.iloc[:,1]
                
            Before_df = pd.DataFrame({'angle': Before_df.iloc[:,0], 'intensity': Intensity_Before}, columns=['angle', 'intensity'])
            After_df = pd.DataFrame({'angle': After_df.iloc[:,0], 'intensity': Intensity_After}, columns=['angle', 'intensity'])
            if Condition:
                Before_df_Plus_1 = pd.DataFrame({'angle': Before_df_Plus_1.iloc[:,0], 'intensity': Intensity_Before_Plus_1}, columns=['angle', 'intensity'])
            plt.plot(Before_df.angle, Before_df.intensity, label='Before',color = 'b')
            if Condition: 
                plt.plot(Before_df_Plus_1.angle, Before_df_Plus_1.intensity, label='Before',color = 'b')
            plt.plot(After_df.angle, After_df.intensity, label='After',color = 'r')
            Normalised_Gaussian_XRD(Before_df, 'Before Ion')
            Normalised_Gaussian_XRD(After_df, 'After Ion')
            if Condition: 
                Normalised_Gaussian_XRD(Before_df_Plus_1, 'Before Ion')
                
            if Select_All == True:
                plt.xlim(Start_Angle, End_Angle)
            plt.legend()
            plt.title(r'XRD: Before and After Ion in {} ({})'.format(Type,Ionisation))
            plt.xlabel("Angle")
            plt.ylabel("Intensity")
            plt.tight_layout()
            
            plt.show()
        
    
        if Look_At_One_XRD == False:
            Data_Numbers_Before = create_array(Max_Data_Number_Before)
            Data_Numbers_After  = create_array(Max_Data_Number_After )
        
    
    
            plt.figure(figsize = (8, 6))
            for Data_Number  in Data_Numbers_Before:
                Before_df = df_Maker_XRD(Type, '', Perovskite, 'Before Ion', Substrate, Test, Version, Data_Number)
        
                if Normalise == True:
                    Intensity_Before = (Before_df.iloc[:,1] - np.min(Before_df.iloc[:,1]))/(np.max(Before_df.iloc[:,1])-np.min(Before_df.iloc[:,1]))
                else:
                    Intensity_Before = Before_df.iloc[:,1]
                Before_df = pd.DataFrame({'angle': Before_df.iloc[:,0], 'intensity': Intensity_Before}, columns=['angle', 'intensity'])
                
                
                plt.plot(Before_df.angle, Before_df.intensity, label='Before',color = 'b')
                Normalised_Gaussian_XRD(Before_df, 'Before Ion')
            
            for Data_Number  in Data_Numbers_After:
                After_df = df_Maker_XRD(Type, Ionisation_XRD, Perovskite, 'After Ion', Substrate, Test, Version, Data_Number)
        
                if Normalise == True:
                    Intensity_After = (After_df.iloc[:,1] - np.min(After_df.iloc[:,1]))/(np.max(After_df.iloc[:,1])-np.min(After_df.iloc[:,1]))
                else:
                    Intensity_After = After_df.iloc[:,1]
                After_df = pd.DataFrame({'angle': After_df.iloc[:,0], 'intensity': Intensity_After}, columns=['angle', 'intensity'])
                plt.plot(After_df.angle, After_df.intensity, label='After',color = 'r')
                Normalised_Gaussian_XRD(After_df, 'After Ion')
                
            for Data_Number  in Data_Numbers_Before:
                Normalised_Gaussian_XRD(Before_df, 'Before Ion')
            for Data_Number  in Data_Numbers_After:
                Normalised_Gaussian_XRD(After_df, 'After Ion')
                
            if Select_All == True:
                plt.xlim(Start_Angle, End_Angle)
            plt.legend()
            plt.title(r'XRD: Before and After Ion in {} ({})'.format(Type,Ionisation))
            plt.xlabel("Angle")
            plt.ylabel("Intensity")
            plt.tight_layout()
            
            plt.show()
    

def Compare_Before_After_Absorbance(
            Type, Ionisation, Version, Colour,
            Start_Wavelength, End_Wavelength,
            Select_All, Normalise):
    
    Before_df_Direct = df_Maker_Absorbance(Type, '', 'Before Ion', 'Direct', Version, Colour)           #Before ionisation dataframe
    Before_df_Indirect = df_Maker_Absorbance(Type, '', 'Before Ion', 'Indirect', Version, Colour)       #Before ionisation dataframe
    After_df_Direct =  df_Maker_Absorbance(Type, Ionisation, 'After Ion', 'Direct', Version, Colour)    #After ionisation dataframe
    After_df_Indirect= df_Maker_Absorbance(Type, Ionisation, 'After Ion', 'Indirect', Version, Colour)  #After ionisation dataframe
    
    Data_Before_Indirect = pd.DataFrame({'wavelength': Before_df_Direct.iloc[:,0], 'intensity': Before_df_Direct.iloc[:,1]}, columns=['wavelength', 'intensity'])
    Data_Before_Direct = pd.DataFrame({'wavelength': Before_df_Indirect.iloc[:,0], 'intensity': Before_df_Indirect.iloc[:,1]}, columns=['wavelength', 'intensity'])
    Data_After_Indirect = pd.DataFrame({'wavelength': After_df_Direct.iloc[:,0], 'intensity': After_df_Direct.iloc[:,1]}, columns=['wavelength', 'intensity'])
    Data_After_Direct = pd.DataFrame({'wavelength': After_df_Indirect.iloc[:,0], 'intensity': After_df_Indirect.iloc[:,1]}, columns=['wavelength', 'intensity'])
        
    if Select_All == False:
        Data_Before_Indirect = select_x_range(Data_Before_Indirect, Start_Wavelength, End_Wavelength)
        Data_Before_Direct   = select_x_range(Data_Before_Direct, Start_Wavelength, End_Wavelength)
        Data_After_Indirect  = select_x_range(Data_After_Indirect, Start_Wavelength, End_Wavelength)
        Data_After_Direct    = select_x_range(Data_After_Direct, Start_Wavelength, End_Wavelength)
    
    Absorbance_Before = absorbance(Data_Before_Direct , Data_Before_Indirect )
    Absorbance_After  = absorbance(Data_After_Direct  , Data_After_Indirect  )
    Wavelength_Before = Data_Before_Direct.wavelength
    Wavelength_After  = Data_After_Direct.wavelength
    
    Absorbance_df_Before = pd.DataFrame({'wavelength': Wavelength_Before, 'intensity': Absorbance_Before}, columns=['wavelength', 'intensity'])
    Absorbance_df_After  = pd.DataFrame({'wavelength': Wavelength_After,  'intensity': Absorbance_After},  columns=['wavelength', 'intensity'])
    
    if Normalise == True:
            Absorbance_Before = (Absorbance_Before - np.min(Absorbance_Before)) /    (np.max(Absorbance_After) -    np.min(Absorbance_Before))
            Absorbance_After =  (Absorbance_After  - np.min(Absorbance_After) ) /    (np.max(Absorbance_After) -    np.min(Absorbance_After) )

    plt.figure(figsize=(8, 6))  
    plt.plot(Absorbance_df_Before.iloc[:,0], Absorbance_Before, label='Before')
    plt.plot(Absorbance_df_After.iloc[:,0], Absorbance_After, label='After')
    plt.legend()
    plt.title(r'Absorption: Before and After Ion in {} ({}) {}'.format(Type,Ionisation, Version))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Absorbance")
    plt.tight_layout()  # Automatically adjust subplot spacing
    
    plt.show()
    
def Compare_Before_After_Absorbance_Intensity(
            Type, Ionisation, Directness, Version, Colour, 
            Start_Wavelength, End_Wavelength,
            Select_All, Normalise):
    
    Before_df = df_Maker_Absorbance(Type, ''        , 'Before Ion', Directness, Version, Colour)           #Before ionisation dataframe
    After_df =  df_Maker_Absorbance(Type, Ionisation, 'After Ion' , Directness, Version, Colour)           #After ionisation dataframe
    
    Data_Before = pd.DataFrame({'wavelength': Before_df.iloc[:,0], 'intensity': Before_df.iloc[:,1]}, columns=['wavelength', 'intensity'])
    Data_After  = pd.DataFrame({'wavelength': After_df.iloc[:,0] , 'intensity': After_df.iloc[:,1]} , columns=['wavelength', 'intensity'])
        
    if Select_All == False:
        Data_Before   = select_x_range(Data_Before, Start_Wavelength, End_Wavelength)
        Data_After    = select_x_range(Data_After , Start_Wavelength, End_Wavelength)
        
    if Normalise == True:
            Data_Before.iloc[:,1] = (Data_Before.iloc[:,1] - np.min(Data_Before.iloc[:,1])) /    (np.max(Data_Before.iloc[:,1]) -    np.min(Data_Before.iloc[:,1]))
            Data_After.iloc[:,1]  = (Data_After.iloc[:,1]  - np.min(Data_After.iloc[:,1]))  /    (np.max(Data_After.iloc[:,1])  -    np.min(Data_After.iloc[:,1]) )

    plt.figure(figsize=(8, 6))  
    plt.plot(Data_Before.iloc[:,0], Data_Before.iloc[:,1], label='Before')
    plt.plot(Data_After.iloc[:,0] ,  Data_After.iloc[:,1], label='After')
    plt.legend()
    plt.title(r'Absorption Intensity ({}): Before and After Ion in {} ({}) {}'.format(Directness, Type, Ionisation, Version))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Absorbance")
    plt.tight_layout()  # Automatically adjust subplot spacing
    plt.show()
     
    
    

def Compare_Before_After_PLQE(
            Type, Ionisation, Version, PLQE_Wavelength, PLQE_Power,
            Start_Wavelength, End_Wavelength, 
            Select_All):

    Before_df_Blank     =    df_Maker_PLQE(Type, '', 'Blank', Version,     'Before Ion', PLQE_Wavelength, PLQE_Power)   #Before ionisation dataframe
    Before_df_Indirect  =    df_Maker_PLQE(Type, '', 'Indirect', Version,  'Before Ion', PLQE_Wavelength, PLQE_Power)   #Before ionisation dataframe
    Before_df_Direct    =    df_Maker_PLQE(Type, '', 'Direct', Version,    'Before Ion', PLQE_Wavelength, PLQE_Power)   #Before ionisation dataframe

    After_df_Blank      =    df_Maker_PLQE(Type, Ionisation, 'Blank', Version,   'After Ion', PLQE_Wavelength, PLQE_Power)   #After ionisation dataframe
    After_df_Indirect   =    df_Maker_PLQE(Type, Ionisation, 'Indirect', Version,'After Ion', PLQE_Wavelength, PLQE_Power)   #After ionisation dataframe
    After_df_Direct     =    df_Maker_PLQE(Type, Ionisation, 'Direct', Version,  'After Ion', PLQE_Wavelength, PLQE_Power)   #After ionisation dataframe

    Wavelength_Before, Intensity_Before_Blank, Intensity_Before_Indirect, Intensity_Before_Direct = Before_df_Blank.iloc[:,0], Before_df_Blank.iloc[:,1], Before_df_Indirect.iloc[:,1], Before_df_Direct.iloc[:,1]
    Wavelength_After , Intensity_After_Blank, Intensity_After_Indirect , Intensity_After_Direct  = After_df_Blank.iloc[:,0] , After_df_Blank.iloc[:,1] , After_df_Indirect.iloc[:,1] , After_df_Direct.iloc[:,1]

    
    ### NORMALISE ###
    if Normalise == True:
        Intensity_Before_Blank      = (Intensity_Before_Blank    - np.min(Intensity_Before_Blank))   /(np.max(Intensity_Before_Blank)   - np.min(Intensity_Before_Blank     ))
        Intensity_Before_Indirect   = (Intensity_Before_Indirect - np.min(Intensity_Before_Indirect))/(np.max(Intensity_Before_Indirect)- np.min(Intensity_Before_Indirect  ))
        Intensity_Before_Direct     = (Intensity_Before_Direct   - np.min(Intensity_Before_Direct))  /(np.max(Intensity_Before_Direct)  - np.min(Intensity_Before_Direct    ))
        Intensity_After_Blank       = (Intensity_After_Blank     - np.min(Intensity_After_Blank))    /(np.max(Intensity_After_Blank)    - np.min(Intensity_After_Blank      ))
        Intensity_After_Indirect    = (Intensity_After_Indirect  - np.min(Intensity_After_Indirect)) /(np.max(Intensity_After_Indirect) - np.min(Intensity_After_Indirect   ))
        Intensity_After_Direct      = (Intensity_After_Direct    - np.min(Intensity_After_Direct))   /(np.max(Intensity_After_Direct)   - np.min(Intensity_After_Direct     ))
    #################
    
    Data_Before_Blank       = pd.DataFrame({'wavelength': Wavelength_Before, 'intensity': Intensity_Before_Blank}   , columns=['wavelength', 'intensity'])
    Data_Before_Indirect    = pd.DataFrame({'wavelength': Wavelength_Before, 'intensity': Intensity_Before_Indirect}, columns=['wavelength', 'intensity'])
    Data_Before_Direct      = pd.DataFrame({'wavelength': Wavelength_Before, 'intensity': Intensity_Before_Direct}  , columns=['wavelength', 'intensity'])
    Data_After_Blank        = pd.DataFrame({'wavelength': Wavelength_After , 'intensity': Intensity_After_Blank}    , columns=['wavelength', 'intensity'])
    Data_After_Indirect     = pd.DataFrame({'wavelength': Wavelength_After , 'intensity': Intensity_After_Indirect} , columns=['wavelength', 'intensity'])
    Data_After_Direct       = pd.DataFrame({'wavelength': Wavelength_After , 'intensity': Intensity_After_Direct}   , columns=['wavelength', 'intensity'])


    if Select_All == False:
        Data_Before_Blank    = select_x_range(Data_Before_Blank     , Start_Wavelength, End_Wavelength)
        Data_Before_Indirect = select_x_range(Data_Before_Indirect  , Start_Wavelength, End_Wavelength)
        Data_Before_Direct   = select_x_range(Data_Before_Direct    , Start_Wavelength, End_Wavelength)
        Data_After_Blank     = select_x_range(Data_After_Blank      , Start_Wavelength, End_Wavelength)
        Data_After_Indirect  = select_x_range(Data_After_Indirect   , Start_Wavelength, End_Wavelength)
        Data_After_Direct    = select_x_range(Data_After_Direct     , Start_Wavelength, End_Wavelength)


    plt.figure(figsize=(8, 6))  
    plt.plot(Data_Before_Blank.wavelength   , Data_Before_Blank.intensity   , label = 'Blank; Before'   )
    plt.plot(Data_After_Blank.wavelength    , Data_After_Blank.intensity    , label = 'Blank; After'    )
    plt.plot(Data_Before_Indirect.wavelength, Data_Before_Indirect.intensity, label = 'Indirect; Before')
    plt.plot(Data_After_Indirect.wavelength , Data_After_Indirect.intensity , label = 'Indirect; After' )
    plt.plot(Data_Before_Direct.wavelength  , Data_Before_Direct.intensity  , label = 'Direct; Before'  )
    plt.plot(Data_After_Direct.wavelength   , Data_After_Direct.intensity   , label = 'Direct; After'   )
    plt.legend()
    plt.title(r'PLQE: Before and After Ion. in {}, ({} {}) '.format(Type,Ionisation,Version))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity")
    plt.tight_layout()  # Automatically adjust subplot spacing
    
    plt.show()
    
def Compare_Before_After_LifeTime(
            Type, Ionisation, WavelengthEm_Before, WavelengthEm_After, WavelengthEx, LifeTime_Version, High_Power_Density_Before, High_Power_Density_After,
            Start_Time, End_Time,
            Select_All, Normalise, Zero_Time, Normalised_Exponential):
    
    Before_df =     df_Maker_LifeTime(Type, ''        , 'Before Ion', WavelengthEx_Before, WavelengthEx, LifeTime_Version, High_Power_Density_Before)    #Before ionisation dataframe
    After_df  =     df_Maker_LifeTime(Type, Ionisation, 'After Ion' , WavelengthEx_After , WavelengthEx, LifeTime_Version, High_Power_Density_After )    #After ionisation dataframe

    Time_Before, Intensity_Before = Before_df.iloc[:,0], Before_df.iloc[:,1]
    Time_After , Intensity_After  = After_df.iloc[:,0] , After_df.iloc[:,1]
    
    
    if Normalise == True:
        ### NORMALISE ###
        Intensity_Before = (Intensity_Before - np.min(Intensity_Before))/(np.max(Intensity_Before)-np.min(Intensity_Before))
        Intensity_After = (Intensity_After - np.min(Intensity_After))/(np.max(Intensity_After)-np.min(Intensity_After))
        #################
    
    Before_df = pd.DataFrame({'time': Time_Before, 'intensity': Intensity_Before}, columns=['time', 'intensity'])
    After_df  = pd.DataFrame({'time': Time_After , 'intensity': Intensity_After} , columns=['time', 'intensity'])

    if Select_All == False:
        Before_df = select_x_range(Before_df, Start_Time, End_Time)
        After_df  = select_x_range(After_df , Start_Time, End_Time)  

    if Zero_Time == True:
        # Find the time corresponding to the maximum intensity
        Max_Time_Before = Before_df.time[Before_df.intensity.idxmax()]
        Max_Time_After = After_df.time[After_df.intensity.idxmax()]
        
        # Find the index of the maximum intensity
        Max_Index_Before = Before_df.intensity.idxmax()
        Max_Index_After = After_df.intensity.idxmax()
        
        # Adjust time axes so that the time of maximum intensity is at zero
        Before_df['time'] -= Max_Time_Before
        After_df['time'] -= Max_Time_After
        
        # Define slicing bounds based on valid ranges
        End_Index_Before = Before_df.index[-1]  # Last valid index of Before_df
        End_Index_After = After_df.index[-1]    # Last valid index of After_df
        
        # Slice DataFrames (ensure inclusive slicing with .loc)
        Before_df = Before_df.loc[Max_Index_Before:End_Index_Before]
        After_df = After_df.loc[Max_Index_After:End_Index_After]
        
        # Debugging output
        print(f"Max_Index_Before: {Max_Index_Before}, End_Index_Before: {End_Index_Before}")
        print(f"Max_Index_After: {Max_Index_After}, End_Index_After: {End_Index_After}")


            
    
    plt.figure(figsize=(8, 6))  
    plt.ylim(0,1)
    plt.plot(Before_df.time, (Before_df.intensity), label='Before')
    plt.plot(After_df.time , (After_df.intensity) , label='After')
    
    if Select_All == False:
        if Normalise == True:
            if Normalised_Exponential == True:
                    Normalised_Exponential_fit(Before_df, 'Before Ion', Predicted_LifeTime, Predicted_Beta, 0.95)
                    Normalised_Exponential_fit(After_df , 'After Ion' , Predicted_LifeTime, Predicted_Beta, 0.85)
                
    plt.legend()
    plt.title(r'Life Time: Before and After Ion in {} ({})'.format(Type,Ionisation))
    plt.xlabel("Time [ns]")
    plt.ylabel("Intensity")
    plt.tight_layout()  # Automatically adjust subplot spacing
    
    plt.show()

def Compare_Before_After_Ex_Em(
            Type, Ionisation, Wavelength_Before, Wavelength_After, Em_Ex,
            Start_Time, End_Time,
            Select_All, Normalise):
    
    Before_df =     df_Maker_Em_Ex(Type, ''        , 'Before Ion', Wavelength_Before, Em_Ex)    #Before ionisation dataframe
    After_df  =     df_Maker_Em_Ex(Type, Ionisation, 'After Ion' , Wavelength_After , Em_Ex)    #After ionisation dataframe

    Time_Before, Intensity_Before = Before_df.iloc[:,0], Before_df.iloc[:,1]
    Time_After , Intensity_After  = After_df.iloc[:,0] , After_df.iloc[:,1]

    if Normalise == True:
        ### NORMALISE ###
        Intensity_Before = (Intensity_Before - np.min(Intensity_Before))/(np.max(Intensity_Before)-np.min(Intensity_Before))
        Intensity_After  = (Intensity_After - np.min(Intensity_After))/(np.max(Intensity_After)-np.min(Intensity_After))
        #################
    
    Before_df = pd.DataFrame({'time': Time_Before, 'intensity': Intensity_Before}, columns=['time', 'intensity'])
    After_df  = pd.DataFrame({'time': Time_After , 'intensity': Intensity_After} , columns=['time', 'intensity'])

    if Select_All == False:
        Before_df = select_x_range(Before_df, Start_Time, End_Time)
        After_df  = select_x_range(After_df , Start_Time, End_Time)

    
    plt.figure(figsize=(8, 6))  
    plt.plot(Before_df.time, (Before_df.intensity), label='Before')
    plt.plot(After_df.time , (After_df.intensity) , label='After')
    plt.legend()
    plt.title(r'{}: Before and After Ion in {} ({})'.format(Em_Ex,Type,Ionisation))
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Intensity")
    plt.tight_layout()  # Automatically adjust subplot spacing
    
    plt.show()
  
def Plot():
    Compare_Before_After_Ex_Em(Type, Ionisation, WavelengthEx_Before, WavelengthEx_After, Ex, Start_Wavelength, End_Wavelength, Select_All, Normalise)   
    Compare_Before_After_Ex_Em(Type, Ionisation, WavelengthEm_Before, WavelengthEm_After, Em, Start_Wavelength, End_Wavelength, Select_All, Normalise)   
    Compare_Before_After_LifeTime(Type, Ionisation, WavelengthEm_Before, WavelengthEm_After, WavelengthEx, LifeTime_Version, High_Power_Density_Before, High_Power_Density_After, Start_Time, End_Time, Select_All, Normalise, Zero_Time, Normalised_Exponential)
    Compare_Before_After_PLQE(Type, Ionisation, Version, PLQE_Wavelength, PLQE_Power, Start_Wavelength, End_Wavelength, Select_All)
    Compare_Before_After_XRD(Type, Ionisation_XRD, Perovskite, Substrate, Version, Test, Chosen_Data_Number_Before, Chosen_Data_Number_After, Start_Angle, End_Angle, Normalise, Select_All, Normalised_Gaussian, Look_At_One_XRD)
    Compare_Before_After_Absorbance(Type, Ionisation, Version, Colour, Start_Wavelength, End_Wavelength, Select_All, Normalise)
    Compare_Before_After_Absorbance_Intensity(Type, Ionisation, 'Direct', Version, Colour, Start_Wavelength, End_Wavelength, Select_All, Normalise)        
    Compare_Before_After_Absorbance_Intensity(Type, Ionisation, 'Indirect', Version, Colour, Start_Wavelength, End_Wavelength, Select_All, Normalise)        

    
Compare_Before_After_XRD(Type, Ionisation_XRD, Perovskite, Substrate, Version, Test,
                         Chosen_Data_Number_Before, Chosen_Data_Number_After, Start_Angle, End_Angle,
                         Normalise, Select_All, Normalised_Gaussian, Look_At_One_XRD, Compare_Peaks, Normal_plots_only)
