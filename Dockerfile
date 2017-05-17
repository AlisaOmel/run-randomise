FROM bids/base_fsl

MAINTAINER John Pellman <john.pellman@childmind.org>

#RUN apt-get update && apt-get install -y wget
#RUN wget -O cpac_install.sh \
    #https://raw.githubusercontent.com/FCP-INDI/C-PAC/0.4.0_development/scripts/cpac_install.sh \
    #&& bash cpac_install.sh

RUN sudo apt-get update && sudo apt-get upgrade -y && apt-get -y install python-dev python-pip 
RUN pip install -ipython 
#RUN pip install NumPy
RUN pip install pyparsing
RUN pip install setuptools
RUN sudo apt-get install -y python-dateutil 
RUN pip install cycler 
RUN pip install pytz
RUN sudo apt-get update && sudo apt-get install -y python-matplotlib
RUN sudo apt-get install -y python-scipy
RUN pip install scikit-learn 
RUN pip install nibabel
RUN sudo apt-get install -y python-nipype
RUN pip install pandas patsy 
RUN pip install -U --user nilearn


ENV FSLDIR /usr/share/fsl/5.0
ENV FSLOUTPUTTYPE NIFTI_GZ
ENV FSLMULTIFILEQUIT TRUE
ENV FSLTCLSH /usr/bin/tclsh
ENV FSLWISH /usr/bin/wish
ENV FSLBROWSER /etc/alternatives/x-www-browser
ENV LD_LIBRARY_PATH /usr/lib/fsl/5.0:${LD_LIBRARY_PATH}
ENV PATH ${FSLDIR}/bin:/home/ubuntu/bin:${PATH} 

COPY create_flame_model_files.py /code/create_flame_model_files.py
COPY run_new.py /code/run_new.py

ENTRYPOINT ["/code/run_new.py"]
