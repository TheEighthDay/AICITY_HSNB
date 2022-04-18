**## post-processing**



We should prepare prediction results(generate from classification) and proposal results(generate from proposal_extract). Prediction results is a folder contains 9 result file. Proposal result is a pkl file.
In total, 4 group prediction results and proposal results are involved.They are "dashboard_128_8","dashboard_128_32","dashboard_32_8","right_128_32".

First, you should modify the path arguements in postprocessing.sh.
\```

    --results_128_8  your_path_to_dashboard_128_8_proposal_file results \
    --results_128_32 your_path_to_dashboard_128_32_proposal_file results \
    --results_32_8 your_path_to_dashboard_32_8_proposal_file results \
    --results_right_128_32 your_path_to_right_128_32_proposal_file results \
    --path_to_fea_right_128_32 your_path_to_right_128_32_prediction_results_folder" \
    --path_to_fea_32_8 your_path_to_dashboard_32_8_prediction_results_folder" \
    --path_to_fea_128_8 your_path_to_dashboard_128_8_prediction_results_folder" \
    --path_to_fea_128_32 your_path_to_dashboard_128_32_prediction_results_folder"

\```

Then, try this command.


\```

sh ./postprocessing.sh

\```

