from pixell import sharp
import sys
sys.path.append('../../tests')
import pixel_tests as ptests
import pickle
import os


import argparse
# Parse command line
parser = argparse.ArgumentParser(description='Generate test results.')
parser.add_argument("-y", "--yaml",     type=str,  default=None,help="Test YAML file.")
parser.add_argument("-o", "--output", type=str,default=None,help='Name of output pkl file.')
args = parser.parse_args()

i = 0
if args.yaml is None:
    yaml_file = ""
    while not(os.path.isfile(yaml_file)) or i==0:
        if i>0: print("File does not exist.")
        yaml_file = input("Enter name of yaml file to generate test results from: ")
        i += 1
else:
    yaml_file = args.yaml
if args.output is None:
    pkl_file = input("Enter name of pickle file to save test results to (exclude extension): ")
else:
    pkl_file = args.output
results,_ = ptests.get_extraction_test_results(yaml_file)
pickle.dump(results,open("%s.pkl" % pkl_file,'wb'),protocol=2)
