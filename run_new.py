#!/usr/bin/env python
import argparse
import os
import sys
import urllib
import pandas as pd
import patsy
import shutil
from create_flame_model_files import create_flame_model_files
import glob
import json
import numpy as np
import subprocess
#import nibabel
#import numpy
#from glob import glob

__version__ = 0.1


def run(command, env={}):
    merged_env = os.environ
    merged_env.update(env)
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, shell=True,
                               env=merged_env)
    while True:
        line = process.stdout.readline()
        line = line.encode('utf-8')[:-1]
        print(line)
        if line == '' and process.poll() != None:
            break
    if process.returncode != 0:
        raise Exception("Non zero return code: %d"%process.returncode)
 
def model_setup(model_file, bids_dir):
	# load in the model 
	with open(model_file) as model_fd:    
		model_dict = json.load(model_fd)

	# parse the model string to determine which columns of the pheno
	# file we are interested in
	incols=model_dict["model"].replace("-1","").replace("-","+").split("+")
	t_incols=[]
	for col in incols:
		if '*' in col:
			t_incols+=col.split("*")
		else:
			t_incols.append(col)
	incols=list(set(t_incols))

	# read in the pheno file
	pheno_df=pd.read_csv(os.path.join(bids_dir, 'participants.tsv'),sep='\t')

	# reduce the file to just the columns that we are interested in
	pheno_df=pheno_df[['participant_id']+incols]

	# remove rows that have empty elements
	pheno_df=pheno_df.dropna()

	# go through data, verify that we can find a corresponding entry in
	# the pheno file, and keep track of the indices so that we can 
	# reorder the pheno to correspond
	file_list=[]
	pheno_key_list=[]
	for root, dirs, files in os.walk(bids_dir):
		for filename in files:
			if not filename.endswith(".nii.gz"):
				continue
			f_chunks = (filename.split(".")[0]).split("_")
			# make a dictionary from the key-value chunks
			f_dict = {chunk.split("-")[0]:"-".join(chunk.split("-")[1:]) for chunk in f_chunks[:-1]}
			pheno_flags=pheno_df["participant_id"]==("-".join(["sub",f_dict["sub"]]))
			if pheno_flags.any():
				pheno_key_list.append(np.where(pheno_flags)[0][0])
				file_list.append(os.path.join(root,filename))

	#### now create the design.mat file

	# remove participant_id column
	pheno_df=pheno_df[incols]

	# reduce to the rows that we are using, and reorder to match the file list
	pheno_df=pheno_df.iloc[pheno_key_list,:]

	print "{0} rows in design matrix".format(len(pheno_df.index))

	#de-mean all numeric columns
	for df_ndx in pheno_df.columns:
		if np.issubdtype(pheno_df[df_ndx].dtype,np.number):
			pheno_df[df_ndx]-=pheno_df[df_ndx].mean()

	# use patsy to create the design matrix
	design=patsy.dmatrix(model_dict["model"],pheno_df,NA_action='raise')
	column_names = design.design_info.column_names

	print 'model terms: {0}'.format(column_names)


	# create contrasts
	if model_dict["contrasts"]:
		contrast_dict={}
		num_contrasts=0
		for k in model_dict["contrasts"]:
			num_contrasts+=1
			try:
			    contrast_dict[k]=[n if n != -0 else 0 for n in design.design_info.linear_constraint(k.encode('ascii')).coefs[0]]
			except patsy.PatsyError as e:
			    if 'token' in e.message:
			        print("A token in contrast \'{0}\' could not be found, should only include tokens from {1}".format(k,column_names))
			    raise
	else:
	    raise ValueError('Model file {0} is missing contrasts'.format(model_file))

	num_subjects=len(file_list)
	mat_file, grp_file, con_file, fts_file = create_flame_model_files(design, \
		column_names, contrast_dict, None, [], None, [1] * num_subjects, "Treatment", \
		"repro_pipe_model", [], working_dir)

	return file_list, num_contrasts, mat_file, con_file

def run_workflow(file_list, working_dir, num_contrasts, mat_file, con_file, num_iterations, output_dir):
	import nipype.pipeline.engine as pe
	import nipype.interfaces.fsl as fsl
	import nipype.interfaces.utility as util
	from nipype.interfaces.afni import preprocess	
	import nipype.interfaces.io as nio

	wf = pe.Workflow(name='wf_randomize')
	wf.base_dir = working_dir
	#merge 
	merge = pe.Node(interface=fsl.Merge(),
					name='fsl_merge')
	merge.inputs.in_files= file_list
	merge.inputs.dimension = 't'
	merge_output = "rando_pipe_merge.nii.gz"
	merge.inputs.merged_file = merge_output 
	
	#mask
	mask = pe.Node(interface= fsl.maths.MathsCommand(), 
				   name = 'fsl_maths')
	mask.inputs.args = '-abs -Tmin -bin'
	merge_mask_output = "rando_pipe_mask.nii.gz"
	mask.inputs.out_file = merge_mask_output
	wf.connect(merge, 'merged_file', mask, 'in_file')

	#randomise
	
	skipTo=[]
	for current_contrast in range(1, num_contrasts+1):
		skipTo.append(' --skipTo={0}'.format(current_contrast))	
	
	randomise = pe.Node (interface= fsl.Randomise(),
		   				name = 'fsl_randomise_{0}'.format(current_contrast))						
	randomise.iterables = ("args", skipTo)
	wf.connect(mask, 'out_file', randomise, 'mask')	
	rando_out_prefix="rando_pipe"
	randomise.inputs.base_name= rando_out_prefix
	randomise.inputs.design_mat = mat_file
	randomise.inputs.tcon = con_file
	randomise.inputs.num_perm = num_iterations
	randomise.inputs.demean = True
	randomise.inputs.tfce = True
	wf.connect(merge, 'merged_file', randomise,'in_file')

	#thresh
	thresh = pe.MapNode (interface = fsl.Threshold(),
				  		 name = 'fsl_threshold_{0}'.format(current_contrast),
				  		 iterfield = ['in_file'])	
	wf.connect(randomise,"t_corrected_p_files", thresh,"in_file")
	thresh.inputs.thresh = 0.95
	thresh_output_file = 'rando_pipe_thresh_tstat{0}'.format(current_contrast)
	thresh.inputs.out_file = thresh_output_file
			
	thresh_bin = pe.MapNode (interface = fsl.maths.MathsCommand(),
					  			 name = 'fsl_threshold_bin_{0}'.format(current_contrast),
					  			 iterfield = ['in_file'])	
	wf.connect(thresh,"out_file", thresh_bin,"in_file")
	thresh.inputs.args = '-bin'

			#ApplyMask
	apply_mask = pe.MapNode(interface = fsl.ApplyMask(),
								name = 'fsl_applymask_{0}'.format(current_contrast),
								iterfield = ['in_file', 'mask_file'])
	wf.connect(randomise, 'tstat_files', apply_mask, 'in_file')	
	wf.connect(thresh_bin, 'out_file', apply_mask, 'mask_file')

			#cluster
	cluster = pe.MapNode(interface = fsl.Cluster(),
					  		 name = 'cluster_{0}'.format(current_contrast),
					  		 iterfield = ['in_file'])	
	wf.connect(apply_mask, 'out_file', cluster, 'in_file')
	cluster.inputs.threshold = 0.0001
	cluster.inputs.out_index_file = "cluster_index_{0}".format(current_contrast)
	cluster.inputs.out_localmax_txt_file = "lmax_{0}.txt".format(current_contrast)
	cluster.inputs.out_size_file = "cluster_size_{0}".format(current_contrast)
	cluster.inputs.out_threshold_file = "rando_out_prefix_{0}".format(current_contrast)	
	cluster.inputs.terminal_output = 'file'	
			
		#save
	datasink = pe.Node(nio.DataSink(), name='sinker_{0}'.format(current_contrast))
	datasink.inputs.base_directory = output_dir
			
	wf.connect(cluster, 'index_file', datasink, 'output.@index_file')
	wf.connect(cluster, 'threshold_file', datasink, 'output.@threshold_file')	
	wf.connect(cluster, 'localmax_txt_file', datasink, 'output.@localmax_txt_file')
	wf.connect(cluster, 'localmax_vol_file', datasink, 'output.@localmax_vol_file')
	wf.connect(cluster, 'max_file', datasink, 'output.@max_file')
	wf.connect(cluster, 'mean_file', datasink,'output.@mean_file')
	wf.connect(cluster, 'pval_file', datasink,'output.@pval_file')
	wf.connect(cluster, 'size_file', datasink,'output.@size_file')
	wf.connect(thresh, 'out_file', datasink, 'output.@thresh_file')	
	wf.run(plugin="MultiProc", plugin_args={"n_procs":10})




parser = argparse.ArgumentParser(description='ABIDE Group Analysis Runner')

parser.add_argument('bids_dir', help='The directory with the input dataset '
                    'formatted according to the BIDS standard.')
parser.add_argument('output_dir', help='The directory where the output files '
                    'should be stored. If you are running group level analysis '
                    'this folder should be prepopulated with the results of the'
                    'participant level analysis.')
parser.add_argument('working_dir', help='The directory where intermediary files '
                    'are stored while working ont them.')
parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                    'Multiple participant level analyses can be run independently '
                    '(in parallel) using the same output_dir. Use test_model to generate'
                    'the model and contrast files, but not run the anlaysis.',
                    choices=['participant', 'group', 'test_model'])
parser.add_argument('model_file', help='JSON file describing the model and contrasts'
                    'that should be.')
parser.add_argument('--num_iterations', help='Number of iterations used by randomise.',
                    default=10, type=int)
#parser.add_argument('--num_processors', help= 'Number of processors used at a time for randomise', default=1, type=int) 
parser.add_argument('--participant_label', help='The label(s) of the participant(s) that should be analyzed. The label '
                   'corresponds to sub-<participant_label> from the BIDS spec '
                   '(so it does not include "sub-"). If this parameter is not '
                   'provided all subjects should be analyzed. Multiple '
                   'participants can be specified with a space separated list.',
                   nargs="+")
parser.add_argument('-v', '--version', action='version',
                    version='BIDS-App example version {}'.format(__version__))
                    

args = parser.parse_args()


model_file=args.model_file
if not os.path.isfile(model_file):
    print("Could not find model file %s"%(model_file))
    sys.exit(1)

output_dir=args.output_dir.rstrip('/')
if not os.path.isdir(output_dir):
    print("Could not find output directory %s"%(output_dir))
    sys.exit(1)

working_dir=args.working_dir.rstrip('/')
if not os.path.isdir(working_dir):
    print("Could not find working directory %s"%(working_dir))
    sys.exit(1)

bids_dir=args.bids_dir.rstrip('/')
if not os.path.isdir(working_dir):
    print("Could not find bids directory %s"%(bids_dir))
    sys.exit(1)

if args.num_iterations:
    num_iterations=int(args.num_iterations)


print ("\n")
print ("## Running randomize pipeline with parameters:")
print ("Output directory: %s"%(bids_dir))
print ("Output directory: %s"%(output_dir))
print ("Working directory: %s"%(working_dir))
print ("Pheno file: %s"%(args.model_file))
print ("Number of iterations: %d"%(num_iterations))
#print ("Number of processors: %d"%(num_processors)
print ("\n")

file_list, num_contrasts, mat_file, con_file = model_setup(model_file, bids_dir)

if args.analysis_level == "participant":
	print("This bids-app does not support individual level analyses")
elif args.analysis_level == "group":
	run_workflow(file_list, working_dir, num_contrasts, mat_file, con_file, num_iterations, output_dir)
