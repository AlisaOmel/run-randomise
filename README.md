# run-randomise

The Docker container contains the environment needed to run the randomisation script and solves many of its dependency issues.

The run-new script uses the FSL randomise tool to analyze fMRI data based on constructed contrasts and models. It has been improved by adopting a nipype workflow and parallelizing independent steps. This code increases the speed of the original run-randomise command. An added feature to the old script allows for automatic data visualization with nilearn. 
