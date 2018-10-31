import pickle



with open(r'C:\Users\walkerl\Documents\ML_tutorial\module0.pkl', "r") as curve_file:
    curves = pickle.load(curve_file)


print(curves)