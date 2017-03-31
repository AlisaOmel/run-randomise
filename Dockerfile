FROM bids/base_fsl

MAINTAINER John Pellman <john.pellman@childmind.org>

#RUN apt-get update && apt-get install -y wget
#RUN wget -O cpac_install.sh \
    #https://raw.githubusercontent.com/FCP-INDI/C-PAC/0.4.0_development/scripts/cpac_install.sh \
    #&& bash cpac_install.sh

RUN sudo apt-get update && sudo apt-get upgrade -y && apt-get -y install python-dev python-pip 
RUN sudo apt-get install -y python-scipy
RUN sudo apt-get install -y python-nipype
RUN pip install pandas patsy 

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
