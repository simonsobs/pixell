from . import pixel_tests as ptests
import pickle
import os

i = 0
yaml_file = ""
while not(os.path.isfile(yaml_file)) or i==0:
    if i>0: print("File does not exist.")
    yaml_file = input("Enter name of yaml file to generate test results from: ")
    i += 1
pkl_file = input("Enter name of pickle file to save test results to (exclude extension): ")
results,_ = ptests.get_extraction_test_results(yaml_file)
pickle.dump(results,open("data/%s.pkl" % pkl_file,'wb'))
