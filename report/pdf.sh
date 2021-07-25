#!/bin/zsh

jupyter nbconvert --to=pdf Two\ Initial\ Methods\ in\ Convolutional\ neural\ network.ipynb -TagRemovePreprocessor.remove_input_tags='{"remove_input"}' -TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' -TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}' --PdfExporter.template_file=./revtex.tplx

#jupyter nbconvert --to pdf Two\ Initial\ Methods\ in\ Convolutional\ neural\ network.ipynb -TagRemovePreprocessor.remove_input_tags='{"remove_input"}' -TagRemovePreprocessor.remove_cell_tags='{"remove_cell"}' -TagRemovePreprocessor.remove_all_outputs_tags='{"remove_output"}' --template=./revtex.tplx