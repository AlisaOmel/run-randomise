#!/usr/bin/env python

import argparse
import os
import sys
import pandas as pd
import patsy
import json
import numpy as np
from create_flame_model_files import create_flame_model_files

__version__ = 0.1


def model_setup(in_model_file, in_bids_dir, model_files_outdir):

    # load in the model
    with open(in_model_file) as model_fd:
        model_dict = json.load(model_fd)

    # parse the model string to determine which columns of the pheno
    # file we are interested in
    in_columns = model_dict["model"].replace("-1", "").replace("-", "+").split("+")
    t_columns = []
    for column in in_columns:
        if '*' in column:
            t_columns += column.split("*")
        else:
            t_columns.append(column)
        in_columns = list(set(t_columns))

    # read in the pheno file
    pheno_df = pd.read_csv(os.path.join(in_bids_dir, 'participants.tsv'), sep='\t')

    # reduce the file to just the columns that we are interested in
    pheno_df = pheno_df[['participant_id'] + in_columns]

    # remove rows that have empty elements
    pheno_df = pheno_df.dropna()

    # go through data, verify that we can find a corresponding entry in
    # the pheno file, and keep track of the indices so that we can
    # reorder the pheno to correspond
    t_file_list = []
    pheno_key_list = []

    for root, dirs, files in os.walk(in_bids_dir):
        for filename in files:

            if not filename.endswith(".nii.gz"):
                continue

            # make a dictionary from the key-value chunks
            f_chunks = (filename.split(".")[0]).split("_")
            f_dict = {chunk.split("-")[0]: "-".join(chunk.split("-")[1:]) for chunk in f_chunks[:-1]}

            if not f_dict['ses']:
                f_dict['ses'] = '1'

            # find the row of the pheno_df that corresponds to the file and save it to pheno_key_list
            participant_index = [index for index, participant_id in enumerate(pheno_df["participant_id"])
                                 if participant_id == "-".join(["sub", f_dict["sub"]])]

            if len(participant_index) == 0:
                print("Could not find entry in phenotype file for {0}, dropping it.".format(
                    os.path.join(root, filename)))
            elif len(participant_index) > 1:
                raise ValueError("Found multiple entries for {0} in {1}", f_participant_id,
                                 os.path.join(in_bids_dir, 'participants.tsv'))
            else:
                pheno_key_list.append(participant_index[0])
                t_file_list.append(os.path.join(root, filename))

    # now create the design.mat file

    # remove participant_id column
    pheno_df = pheno_df[in_columns]

    # reduce to the rows that we are using, and reorder to match the file list
    pheno_df = pheno_df.iloc[pheno_key_list, :]

    print "{0} rows in design matrix".format(len(pheno_df.index))

    # de-mean all numeric columns, we expect categorical variables to be encoded with strings
    for df_ndx in pheno_df.columns:
        if np.issubdtype(pheno_df[df_ndx].dtype, np.number):
            pheno_df[df_ndx] -= pheno_df[df_ndx].mean()

    # use patsy to create the design matrix
    design = patsy.dmatrix(model_dict["model"], pheno_df, NA_action='raise')
    column_names = design.design_info.column_names

    print('Model terms: {0}'.format(', '.join(column_names)))

    # create contrasts
    if model_dict["contrasts"]:
        contrast_dict = {}
        t_num_contrasts = 0

        for k in model_dict["contrasts"]:
            t_num_contrasts += 1
            try:
                contrast_dict[k] = [n if n != -0 else 0
                                    for n in design.design_info.linear_constraint(k.encode('ascii')).coefs[0]]
            except patsy.PatsyError as e:
                if 'token' in e.message:
                    print("A token in contrast \'{0}\' could not be found, should only include tokens from {1}".format(
                        k, ', '.join(column_names)))
                raise
    else:
        raise ValueError('Model file {0} is missing contrasts'.format(model_file))

    num_subjects = len(t_file_list)
    t_mat_file, t_grp_file, t_con_file, t_fts_file = create_flame_model_files(design, column_names,
                                                                              contrast_dict, None, [],
                                                                              None, [1] * num_subjects,
                                                                              "Treatment",
                                                                              "randomise_pipe_model",
                                                                              [], model_files_outdir)

    return t_file_list, t_num_contrasts, t_mat_file, t_con_file


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='ABIDE Group Analysis Runner')

    parser.add_argument('bids_dir', help='The directory with the input dataset '
                        'formatted according to the BIDS standard.')
    parser.add_argument('output_dir', help='The directory where the output files '
                        'should be stored. If you are running group level analysis '
                        'this folder should be prepopulated with the results of the'
                        'participant level analysis.')
    parser.add_argument('working_dir', help='The directory where intermediary files '
                        'are stored while working on them.')
    parser.add_argument('analysis_level', help='Level of the analysis that will be performed. '
                        'Multiple participant level analyses can be run independently '
                        '(in parallel) using the same output_dir. Use test_model to generate'
                        'the model and contrast files, but not run the anlaysis.',
                        choices=['participant', 'group', 'test_model'])
    parser.add_argument('model_file', help='JSON file describing the model and contrasts'
                        'that should be.')
    parser.add_argument('--num_iterations', help='Number of iterations used by randomise.',
                        default=10000, type=int)
    parser.add_argument('--num_processors', help='Number of processors used at a time for randomise',
                        default=2, type=int)
    parser.add_argument('-v', '--version', action='version',
                        version='BIDS-App example version {}'.format(__version__))

    args = parser.parse_args()

    model_file = args.model_file
    if not os.path.isfile(model_file):
        print("Could not find model file {0}".format(model_file))
        sys.exit(1)

    output_dir = args.output_dir.rstrip('/')
    if not os.path.isdir(output_dir):
        print("Could not find output directory {0}".format(output_dir))
        sys.exit(1)

    working_dir = args.working_dir.rstrip('/')
    if not os.path.isdir(working_dir):
        print("Could not find working directory {0}".format(working_dir))
        sys.exit(1)

    bids_dir = args.bids_dir.rstrip('/')
    if not os.path.isdir(working_dir):
        print("Could not find bids directory {0}".format(bids_dir))
        sys.exit(1)

    num_iterations = 10000
    if args.num_iterations:
        num_iterations = int(args.num_iterations)

    num_processors = 1
    if args.num_processors:
        num_processors = int(args.num_processors)

    print ("\n")
    print ("## Running randomize pipeline with parameters:")
    print ("Output directory: {0}".format(bids_dir))
    print ("Output directory: {0}".format(output_dir))
    print ("Working directory: {0}".format(working_dir))
    print ("Pheno file: {0}".format(args.model_file))
    print ("Number of iterations: {0}".format(num_iterations))
    print ("Number of processors: {0}".format(num_processors))
    print ("\n")

    file_list, num_contrasts, mat_file, con_file = model_setup(model_file, bids_dir, working_dir)

    if args.analysis_level == "participant":
        print("This bids-app does not support individual level analyses")

    elif args.analysis_level == "group":

        import nipype.pipeline.engine as pe
        import nipype.interfaces.fsl as fsl
        import nipype.interfaces.io as nio
        import nipype.interfaces.utility as niu
        wf = pe.Workflow(name='wf_randomize')
        wf.base_dir = working_dir

        # First merge input files into single 4D file
        merge = pe.Node(interface=fsl.Merge(), name='fsl_merge')
        merge.inputs.in_files = file_list
        merge.inputs.dimension = 't'
        merge_output = "randomise_pipe_merge.nii.gz"
        merge.inputs.merged_file = merge_output

        # Create a mask from the merged file
        mask = pe.Node(interface=fsl.maths.MathsCommand(), name='fsl_maths')
        mask.inputs.args = '-abs -Tmin -bin'
        merge_mask_output = "randomise_pipe_mask.nii.gz"
        mask.inputs.out_file = merge_mask_output
        wf.connect(merge, 'merged_file', mask, 'in_file')

        # We want to parallelize so that each contrast is processed
        # separately
        def select(input_list):
            out_file = input_list[0]
            return out_file

        for current_contrast in range(1, num_contrasts + 1):
            # use randomize to use perform permutation test for contrast
            randomise = pe.Node(interface=fsl.Randomise(), name='fsl_randomise_{0}'.format(current_contrast))
            wf.connect(mask, 'out_file', randomise, 'mask')
            randomise.inputs.base_name = "randomise_pipe_contrast_{0}".format(current_contrast)
            randomise.inputs.design_mat = mat_file
            randomise.inputs.tcon = con_file
            randomise.inputs.args = ' --skipTo={0}'.format(current_contrast)
            randomise.inputs.num_perm = num_iterations
            randomise.inputs.demean = True
            randomise.inputs.tfce = True
            wf.connect(merge, 'merged_file', randomise, 'in_file')

            select_t_corrected = pe.Node(niu.Function(input_names=["input_list"],
                                               output_names=['out_file'],
                                               function=select),
                                               name = 'select_t_cor{0}'.format(current_contrast))

            wf.connect(randomise, "t_corrected_p_files", select_t_corrected, "input_list")


            # threshold the resulting t corrected p file
            thresh = pe.Node(interface=fsl.Threshold(),
                             name='fsl_threshold_contrast_{0}'.format(current_contrast))
            thresh.inputs.thresh = 0.95
            wf.connect(select_t_corrected, "out_file", thresh, "in_file")
            thresh_output_file = 'rando_pipe_thresh_tstat{0}.nii.gz'.format(current_contrast)
            thresh.inputs.out_file = thresh_output_file

            # binarize the result of applying the threshold to get a mask
            thresh_bin = pe.Node(interface=fsl.maths.MathsCommand(),
                                 name='fsl_threshold_bin_contrast_{0}'.format(current_contrast))
            thresh_bin.inputs.args = '-bin'
            wf.connect(thresh, "out_file", thresh_bin, "in_file")

            select_t_stat = pe.Node(niu.Function(input_names=["input_list"],
                                                      output_names=['out_file'],
                                                      function=select),
                                         name='select_item_t_stat{0}'.format(current_contrast))

            wf.connect(randomise, "tstat_files", select_t_stat, "input_list")

            # apply calculated mask to the statistic image
            apply_mask = pe.Node(interface=fsl.ApplyMask(),
                                 name='fsl_applymask_contrast_{0}'.format(current_contrast))
            wf.connect(select_t_stat, 'out_file', apply_mask, 'in_file')
            wf.connect(thresh_bin, 'out_file', apply_mask, 'mask_file')

            # cluster the results to get a report of the findings
            cluster = pe.Node(interface=fsl.Cluster(),
                              name='cluster_contrast_{0}'.format(current_contrast))
            cluster.inputs.threshold = 0.0001
            cluster.inputs.out_index_file = "cluster_index_contrast_{0}".format(current_contrast)
            cluster.inputs.out_localmax_txt_file = "lmax_contrast_{0}.txt".format(current_contrast)
            cluster.inputs.out_size_file = "cluster_size_contrast_{0}".format(current_contrast)
            cluster.inputs.out_threshold_file = "randomise_out_contrast_{0}".format(current_contrast)
            cluster.inputs.terminal_output = 'file'
            wf.connect(apply_mask, 'out_file', cluster, 'in_file')

            # attach a datasink to save the output
            datasink = pe.Node(nio.DataSink(), name='sinker_contrast_{0}'.format(current_contrast))
            datasink.inputs.base_directory = output_dir

            wf.connect(apply_mask, 'out_file', datasink, 'output.@thresh_stat_file')
            wf.connect(cluster, 'index_file', datasink, 'output.@index_file')
            wf.connect(cluster, 'threshold_file', datasink, 'output.@threshold_file')
            wf.connect(cluster, 'localmax_txt_file', datasink, 'output.@localmax_txt_file')
            wf.connect(cluster, 'localmax_vol_file', datasink, 'output.@localmax_vol_file')
            wf.connect(cluster, 'max_file', datasink, 'output.@max_file')
            wf.connect(cluster, 'mean_file', datasink, 'output.@mean_file')
            wf.connect(cluster, 'pval_file', datasink, 'output.@pval_file')
            wf.connect(cluster, 'size_file', datasink, 'output.@size_file')

        wf.run(plugin="MultiProc", plugin_args={"n_procs": num_processors})
