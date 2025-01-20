# gpt-2-on-fpga
This repository consist of gpt-2 model implementation on fpga board:

1) Pre-trained model has been leverages and done using Python in the Google-Colab & related (.ipynb) file provided in the gpt-2 selection folder.

2) Basic building blocks of GPT-2 decoder blocks are provided as a Hardware repository. Consist of MHA block, ffn and tokenizer block. The obtained high-level description of the pre-trained model and later, transform into RTL logic through the HLS process using the Vitis-HLS compiler and the obtained RTL are provided in the RTL and IP integration folder.

3) The PYNQ repository consist of overlay (driver) python script and other relevant files (bit, hwh) are given in overlay files of the pynq-overlay repository.
   
NOTE: 1) Each (.ipynb) file consists of software results, driver file and ARM-CPU result. Additionally, all C++ files consist of different GPT-2 blocks code from scratch.

In case of any inconvenience drop a mail to respective email id: m23eev006@iitj.ac.in, yadav.49@iitj.ac.in
